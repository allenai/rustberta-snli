use anyhow::Result;
use crossbeam::thread;
use log::debug;
use rust_tokenizers::tokenizer::{RobertaTokenizer, Tokenizer, TruncationStrategy};
use rust_tokenizers::TokenizedInput;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufRead};
use tch::Tensor;

use crate::common;
use crate::tokenization::{load_tokenizer, PAD_TOKEN_ID};

#[derive(Debug)]
pub struct Batch {
    pub token_ids: Tensor,
    pub type_ids: Tensor,
    pub gold_labels: Option<Tensor>,
    pub mask: Tensor,
}

impl Batch {
    pub fn new(token_ids: &[i64], type_ids: &[i8], gold_label: Option<u8>) -> Self {
        Self {
            token_ids: Tensor::of_slice(token_ids).unsqueeze(0),
            type_ids: Tensor::of_slice(type_ids)
                .totype(tch::Kind::Int64)
                .unsqueeze(0),
            gold_labels: gold_label.map(|label| Tensor::from(label).unsqueeze(0)),
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

    pub fn combine(batches: &[Batch]) -> Self {
        if batches.is_empty() {
            panic!("Tried to combine an empty slice of batches");
        }

        // Find max sequence length.
        let mut max_len = 0;
        batches
            .iter()
            .for_each(|b| max_len = std::cmp::max(max_len, b.token_ids.size()[1]));

        let mut token_ids_tensors: Vec<Tensor> = Vec::with_capacity(batches.len());
        let mut type_ids_tensors: Vec<Tensor> = Vec::with_capacity(batches.len());
        let mut gold_labels_tensors: Option<Vec<&Tensor>> = match batches[0].gold_labels {
            Some(_) => Some(Vec::with_capacity(batches.len())),
            _ => None,
        };
        let mut mask_tensors: Vec<Tensor> = Vec::with_capacity(batches.len());

        for b in batches {
            let current_len = b.token_ids.size()[1];
            let padding_needed = (max_len - current_len) as usize;

            if padding_needed > 0 {
                let token_ids = Tensor::cat(
                    &[
                        &b.token_ids,
                        &Tensor::of_slice(&vec![PAD_TOKEN_ID; padding_needed]).unsqueeze(0),
                    ],
                    1,
                );
                token_ids_tensors.push(token_ids);

                let type_ids = Tensor::cat(
                    &[
                        &b.type_ids,
                        &Tensor::of_slice(&vec![PAD_TOKEN_ID; padding_needed]).unsqueeze(0),
                    ],
                    1,
                );
                type_ids_tensors.push(type_ids);

                let mask = Tensor::cat(
                    &[
                        &b.mask,
                        &Tensor::of_slice(&vec![0; padding_needed]).unsqueeze(0),
                    ],
                    1,
                );
                mask_tensors.push(mask);
            } else {
                token_ids_tensors.push(b.token_ids.copy());
                type_ids_tensors.push(b.type_ids.copy());
                mask_tensors.push(b.mask.copy());
            }

            if let Some(label_tensors) = gold_labels_tensors.as_mut() {
                label_tensors.push(b.gold_labels.as_ref().unwrap());
            } else if b.gold_labels.is_some() {
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

pub struct Reader {
    tokenizer: RobertaTokenizer,
    max_sequence_length: usize,
    truncation_strategy: TruncationStrategy,
    n_workers: usize,
}

impl Reader {
    pub fn new(model_resource_dir: &str) -> Result<Self> {
        Ok(Reader {
            tokenizer: load_tokenizer(model_resource_dir)?,
            truncation_strategy: TruncationStrategy::LongestFirst,
            max_sequence_length: 512,
            n_workers: num_cpus::get(),
        })
    }

    pub fn read(&self, path: &str) -> Result<Vec<Batch>> {
        let (tx_line, rx_line) = flume::unbounded::<String>();
        let (tx_batch, rx_batch) = flume::unbounded::<Batch>();

        thread::scope(|s| {
            let mut workers: Vec<thread::ScopedJoinHandle<_>> = Vec::with_capacity(self.n_workers);
            for i in 0..self.n_workers {
                let rx_line = rx_line.clone();
                let tx_batch = tx_batch.clone();
                let handle = s.spawn(move |_| {
                    debug!("Worker[{}] initializing", i);
                    self.process_lines(rx_line, tx_batch);
                    debug!("Worker[{}] finished", i);
                });
                workers.push(handle);
            }

            let path = String::from(path);
            let producer = s.spawn(move |_| {
                let file = File::open(path).expect("Failed to read file");
                let lines = io::BufReader::new(file).lines();
                for line in lines {
                    let line = line.expect("IO error reading line");
                    tx_line
                        .send(line)
                        .expect("Failed to send line through channel");
                }
            });

            drop(tx_batch);

            let progress_bar = common::new_progress_bar();
            let data: Vec<Batch> = progress_bar.wrap_iter(rx_batch.iter()).collect();
            progress_bar.finish_at_current_pos();

            producer.join().expect("The producer has panicked");
            for worker in workers {
                worker.join().expect("The worker has panicked");
            }

            Ok(data)
        })
        .unwrap()
    }

    fn process_lines(&self, rx_line: flume::Receiver<String>, tx_batch: flume::Sender<Batch>) {
        rx_line.iter().for_each(|line| {
            let instance: Instance =
                serde_json::from_str(&line).expect("Failed to deserialize instance");
            match instance.gold_label.as_deref() {
                Some("-") => {
                    // invalid instance, skip.
                }
                _ => {
                    let inputs = self.tokenizer.encode(
                        &instance.premise,
                        Some(&instance.hypothesis),
                        self.max_sequence_length,
                        &self.truncation_strategy,
                        0,
                    );
                    let batch =
                        Batch::from_tokenized_input(&inputs, instance.gold_label.as_deref());

                    tx_batch
                        .send(batch)
                        .expect("Failed to send batch through channel");
                }
            };
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batching() {
        let batches = vec![
            Batch::new(&[0, 3, 4, 2], &[0, 0, 0, 0], Some(0)),
            Batch::new(&[0, 3, 2], &[0, 0, 0], Some(1)),
        ];
        assert_eq!(batches[0].size(), (1, 4));
        assert_eq!(batches[1].size(), (1, 3));

        let batch = Batch::combine(&batches);
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
        let mut reader = Reader::new("test_fixtures/tokenizer").unwrap();
        // Set 1 worker so the order is deterministic.
        reader.n_workers = 1;

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
