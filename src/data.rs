use anyhow::Result;
use log::info;
use rust_tokenizers::{TokenizedInput, Tokenizer, TruncationStrategy};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufRead};
use std::thread;
use tch::Tensor;

use crate::common;

#[derive(Debug)]
pub struct Batch {
    pub token_ids: Tensor,
    pub type_ids: Tensor,
    pub gold_label: Option<Tensor>,
}

impl Batch {
    pub fn new(token_ids: &[i64], type_ids: &[i8], gold_label: Option<u8>) -> Self {
        Self {
            token_ids: Tensor::of_slice(token_ids).unsqueeze(0),
            type_ids: Tensor::of_slice(type_ids)
                .totype(tch::Kind::Int64)
                .unsqueeze(0),
            gold_label: gold_label.map(|label| Tensor::from(label).unsqueeze(0)),
        }
    }

    pub fn from_tokenized_input(input: &TokenizedInput, gold_label: Option<&str>) -> Self {
        Self::new(
            &input.token_ids,
            &input.segment_ids,
            gold_label.map(|label| common::label2id(label)),
        )
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
    tokenizer_dir: String,
    max_sequence_length: usize,
    n_workers: usize,
}

impl Reader {
    pub fn new(model_resource_dir: &str) -> Result<Self> {
        Ok(Reader {
            tokenizer_dir: String::from(model_resource_dir),
            max_sequence_length: 512,
            n_workers: num_cpus::get_physical(),
        })
    }

    pub fn read(&self, path: &str) -> Result<Vec<Batch>> {
        let (tx_line, rx_line) = flume::unbounded::<String>();
        let (tx_batch, rx_batch) = flume::unbounded::<Batch>();

        let mut workers: Vec<thread::JoinHandle<_>> = Vec::with_capacity(self.n_workers);
        for i in 0..self.n_workers {
            let rx_line = rx_line.clone();
            let tx_batch = tx_batch.clone();

            let tokenizer_dir = self.tokenizer_dir.clone();
            let max_sequence_length = self.max_sequence_length;

            let handle = thread::spawn(move || {
                info!("Worker[{}] initializing", i);
                let tokenizer = crate::tokenization::load_tokenizer(tokenizer_dir)
                    .expect("failed to load tokenizer");

                rx_line.iter().for_each(|line| {
                    let instance: Instance =
                        serde_json::from_str(&line).expect("Failed to deserialize instance");
                    match instance.gold_label.as_deref() {
                        Some("-") => {
                            // invalid instance, skip.
                        }
                        _ => {
                            let inputs = tokenizer.encode(
                                &instance.premise,
                                Some(&instance.hypothesis),
                                max_sequence_length,
                                &TruncationStrategy::LongestFirst,
                                0,
                            );
                            let batch = Batch::from_tokenized_input(
                                &inputs,
                                instance.gold_label.as_deref(),
                            );

                            tx_batch
                                .send(batch)
                                .expect("Failed to send batch through channel");
                        }
                    };
                });

                info!("Worker[{}] finished", i);
            });

            workers.push(handle);
        }

        let path = String::from(path);
        let producer = thread::spawn(move || {
            let file = File::open(path).expect("Failed to read fiel");
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reader() {
        let mut reader = Reader::new("test_fixtures/tokenizer").unwrap();
        // Set 1 worker so the order is deterministic.
        reader.n_workers = 1;

        let tokenizer = crate::tokenization::load_tokenizer(&reader.tokenizer_dir)
            .expect("failed to load tokenizer");

        let mut data = reader.read("test_fixtures/snli.jsonl").unwrap();

        assert_eq!(data.len(), 3);

        let batch = data.remove(0);

        assert!(batch.gold_label.is_some());
        let label = batch.gold_label.as_ref().unwrap();
        assert_eq!(label.size(), vec![1]);
        assert_eq!(Vec::<i64>::from(label), [2]);

        let token_ids = Vec::<i64>::from(batch.token_ids);
        let string = tokenizer.decode(
            token_ids, false, // skip special tokens
            true,  // clean up tokenizaiton spaces
        );
        assert_eq!(
            string.as_str(),
            "<s> A person on a horse jumps over a broken down airplane.</s></s> A person is training his horse for a competition.</s>"
        )
    }
}
