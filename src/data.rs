use anyhow::Result;
use crossbeam::thread;
use log::debug;
use rust_tokenizers::tokenizer::{RobertaTokenizer, Tokenizer, TruncationStrategy};
use rust_tokenizers::TokenizedInput;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::sync::RwLock;
use tch::Tensor;

use crate::common;
use crate::tokenization::{load_tokenizer, PAD_TOKEN_ID};

#[derive(Debug)]
pub struct Batch {
    /// Shape: (batch_size, sequence length)
    pub token_ids: Tensor,
    /// Shape: (batch_size, sequence length)
    pub type_ids: Tensor,
    /// Shape: (batch_size,)
    pub gold_labels: Option<Tensor>,
    /// Shape: (batch_size, sequence length)
    pub mask: Tensor,
}

impl Batch {
    pub fn new(token_ids: &[i64], type_ids: &[i8], gold_label: Option<u8>) -> Self {
        Self {
            token_ids: Tensor::of_slice(token_ids).unsqueeze(0),
            type_ids: Tensor::of_slice(type_ids)
                .totype(tch::Kind::Int64)
                .unsqueeze(0),
            gold_labels: gold_label
                .map(|label| Tensor::from(label).totype(tch::Kind::Int64).unsqueeze(0)),
            mask: Tensor::of_slice(&vec![1; token_ids.len()]).unsqueeze(0),
        }
    }

    pub fn to_device(&self, device: tch::Device) -> Self {
        Self {
            token_ids: self.token_ids.to_device(device),
            type_ids: self.type_ids.to_device(device),
            gold_labels: self
                .gold_labels
                .as_ref()
                .map(|label| label.to_device(device)),
            mask: self.mask.to_device(device),
        }
    }

    pub fn from_tokenized_input(input: &TokenizedInput, gold_label: Option<&str>) -> Self {
        Self::new(
            &input.token_ids,
            &input.segment_ids,
            gold_label.map(|label| common::label2id(label)),
        )
    }

    pub fn size(&self) -> (i64, i64) {
        let size = self.token_ids.size();
        (size[0], size[1])
    }

    pub fn combine<B: std::borrow::Borrow<Batch>>(batches: &[B]) -> Self {
        if batches.is_empty() {
            panic!("Tried to combine an empty slice of batches");
        }

        // Find max sequence length.
        let mut max_len = 0;
        batches
            .iter()
            .for_each(|b| max_len = std::cmp::max(max_len, b.borrow().token_ids.size()[1]));

        let mut token_ids_tensors: Vec<Tensor> = Vec::with_capacity(batches.len());
        let mut type_ids_tensors: Vec<Tensor> = Vec::with_capacity(batches.len());
        let mut gold_labels_tensors: Option<Vec<&Tensor>> = match batches[0].borrow().gold_labels {
            Some(_) => Some(Vec::with_capacity(batches.len())),
            _ => None,
        };
        let mut mask_tensors: Vec<Tensor> = Vec::with_capacity(batches.len());

        for b in batches {
            let current_len = b.borrow().token_ids.size()[1];
            let padding_needed = (max_len - current_len) as usize;

            if padding_needed > 0 {
                let token_ids = Tensor::cat(
                    &[
                        &b.borrow().token_ids,
                        &Tensor::of_slice(&vec![PAD_TOKEN_ID; padding_needed]).unsqueeze(0),
                    ],
                    1,
                );
                token_ids_tensors.push(token_ids);

                let type_ids = Tensor::cat(
                    &[
                        &b.borrow().type_ids,
                        &Tensor::of_slice(&vec![PAD_TOKEN_ID; padding_needed]).unsqueeze(0),
                    ],
                    1,
                );
                type_ids_tensors.push(type_ids);

                let mask = Tensor::cat(
                    &[
                        &b.borrow().mask,
                        &Tensor::of_slice(&vec![0; padding_needed]).unsqueeze(0),
                    ],
                    1,
                );
                mask_tensors.push(mask);
            } else {
                token_ids_tensors.push(b.borrow().token_ids.copy());
                type_ids_tensors.push(b.borrow().type_ids.copy());
                mask_tensors.push(b.borrow().mask.copy());
            }

            if let Some(label_tensors) = gold_labels_tensors.as_mut() {
                label_tensors.push(b.borrow().gold_labels.as_ref().unwrap());
            } else if b.borrow().gold_labels.is_some() {
                panic!("expect batch without gold labels");
            }
        }

        Batch {
            token_ids: Tensor::cat(&token_ids_tensors, 0),
            type_ids: Tensor::cat(&type_ids_tensors, 0),
            gold_labels: gold_labels_tensors.map(|labels| Tensor::cat(&labels, 0)),
            mask: Tensor::cat(&mask_tensors, 0),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Instance {
    #[serde(alias = "sentence1")]
    premise: String,
    #[serde(alias = "sentence2")]
    hypothesis: String,
    gold_label: Option<String>,
}

impl Instance {
    pub fn new(premise: &str, hypothesis: &str, gold_label: Option<&str>) -> Self {
        Self {
            premise: premise.into(),
            hypothesis: hypothesis.into(),
            gold_label: gold_label.map(String::from),
        }
    }
}

pub struct Reader {
    pub tokenizer: RobertaTokenizer,
    pub max_sequence_length: usize,
    pub truncation_strategy: TruncationStrategy,
    pub max_instances: Option<usize>,
    num_workers: usize,
    is_done: RwLock<bool>,
}

impl Reader {
    pub fn new<P: AsRef<Path>>(vocab_path: P, merges_path: P) -> Result<Self> {
        Ok(Reader {
            tokenizer: load_tokenizer(vocab_path, merges_path)?,
            truncation_strategy: TruncationStrategy::LongestFirst,
            max_sequence_length: 512,
            num_workers: std::cmp::min(4, num_cpus::get()),
            max_instances: None,
            is_done: RwLock::new(false),
        })
    }

    pub fn encode_instance(&self, instance: &Instance) -> Batch {
        let inputs = self.tokenizer.encode(
            &instance.premise,
            Some(&instance.hypothesis),
            self.max_sequence_length,
            &self.truncation_strategy,
            0,
        );
        Batch::from_tokenized_input(&inputs, instance.gold_label.as_deref())
    }

    pub fn read(&self, path: &str) -> Result<Vec<Batch>> {
        self.mark_done(false);

        let (tx, rx) = flume::unbounded::<Batch>();
        let path = String::from(path);

        thread::scope(|s| {
            let mut workers: Vec<thread::ScopedJoinHandle<_>> =
                Vec::with_capacity(self.num_workers);

            for i in 0..self.num_workers {
                let tx = tx.clone();
                let path = path.clone();
                let handle = s.spawn(move |_| {
                    debug!("Worker[{}] initialized", i);

                    let file = File::open(path).expect("Failed to read file");
                    let lines = io::BufReader::new(file).lines();
                    for (n, line) in lines.skip(i).step_by(self.num_workers).enumerate() {
                        if let Some(max_instances) = self.max_instances {
                            if n * self.num_workers >= max_instances {
                                // We might be done.
                                if *self.is_done.read().expect("Failed to get read lock") {
                                    break;
                                }
                            }
                        }

                        let line = line.expect("IO error reading line");
                        let instance: Instance =
                            serde_json::from_str(&line).expect("Failed to deserialize instance");

                        match instance.gold_label.as_deref() {
                            Some("-") => {
                                // invalid instance, skip.
                            }
                            _ => {
                                let batch = self.encode_instance(&instance);
                                tx.send(batch).ok();
                            }
                        };
                    }

                    debug!("Worker[{}] finished", i);
                });
                workers.push(handle);
            }

            drop(tx);

            debug!("Collecting instances from workers");
            let progress_bar = common::new_spinner();
            let data: Vec<Batch>;
            if let Some(max_instances) = self.max_instances {
                data = progress_bar
                    .wrap_iter(rx.iter().take(max_instances))
                    .collect();
                self.mark_done(true);
                drop(rx);
            } else {
                data = progress_bar.wrap_iter(rx.iter()).collect();
            }
            progress_bar.finish_at_current_pos();

            debug!("Finished, waiting for workers to shutdown");
            for worker in workers {
                worker.join().expect("The worker has panicked");
            }
            debug!("Done!");

            Ok(data)
        })
        .unwrap()
    }

    fn mark_done(&self, is_done: bool) {
        let mut done = self.is_done.write().expect("Faild to get write lock");
        *done = is_done;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batching() {
        let x1 = Batch::new(&[0, 3, 4, 2], &[0, 0, 0, 0], Some(0));
        let x2 = Batch::new(&[0, 3, 2], &[0, 0, 0], Some(1));
        let batches = vec![&x1, &x2];
        assert_eq!(batches[0].size(), (1, 4));
        assert_eq!(batches[1].size(), (1, 3));

        let batch = Batch::combine(&batches[..]);
        assert_eq!(batch.size(), (2, 4));

        assert_eq!(
            Vec::<Vec<i64>>::from(batch.token_ids),
            vec![[0, 3, 4, 2], [0, 3, 2, 1]]
        );
        assert_eq!(
            Vec::<Vec<i64>>::from(batch.type_ids),
            vec![[0, 0, 0, 0], [0, 0, 0, 1]]
        );
        assert_eq!(
            Vec::<i64>::from(batch.gold_labels.as_ref().unwrap()),
            vec![0, 1]
        );
        assert_eq!(
            Vec::<Vec<i64>>::from(batch.mask),
            vec![[1, 1, 1, 1], [1, 1, 1, 0]]
        );
    }

    #[test]
    fn test_reader() {
        let mut reader = Reader::new(
            "test_fixtures/tokenizer/vocab.txt",
            "test_fixtures/tokenizer/merges.txt",
        )
        .unwrap();
        // Set 1 worker so the order is deterministic.
        reader.num_workers = 1;

        let mut data = reader.read("test_fixtures/snli.jsonl").unwrap();

        assert_eq!(data.len(), 3);

        let batch = data.remove(0);

        assert!(batch.gold_labels.is_some());
        let label = batch.gold_labels.as_ref().unwrap();
        assert_eq!(label.size(), vec![1]);
        assert_eq!(Vec::<i64>::from(label), [2]);

        let token_ids = Vec::<i64>::from(batch.token_ids);
        let string = reader.tokenizer.decode(
            token_ids, false, // skip special tokens
            true,  // clean up tokenizaiton spaces
        );
        assert_eq!(
            string.as_str(),
            "<s> A person on a horse jumps over a broken down airplane.</s></s> A person is training his horse for a competition.</s>"
        )
    }
}
