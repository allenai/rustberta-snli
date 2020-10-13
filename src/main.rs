use anyhow::Result;
use cached_path::{self, cached_path_with_options};
use env_logger::Env;
use log::info;
use rust_tokenizers::{Tokenizer, TruncationStrategy};
use structopt::StructOpt;
use tch::Device;

pub(crate) mod common;
pub mod data;
pub mod modeling;
pub mod tokenization;

use crate::modeling::TransformerSequenceClassificationModel;

const TRANSFORMER_MODEL: &str =
    "https://storage.googleapis.com/allennlp-public-models/rustberta.tar.gz";
const TRAIN_PATH: &str = "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_train.jsonl";
const DEV_PATH: &str = "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_dev.jsonl";
// const TEST_PATH: &str = "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_test.jsonl";
const MAX_SEQUENCE_LENGTH: usize = 512;
const TRUNCATION_STRATEGY: TruncationStrategy = TruncationStrategy::LongestFirst;

fn main() -> Result<()> {
    env_logger::from_env(Env::default().default_filter_or("info")).init();
    let opt = RustBERTaOpt::from_args();

    match opt {
        RustBERTaOpt::Train => {
            info!("Training RustBERTa on SNLI");
            train()?;
        }
        RustBERTaOpt::Evaluate => {
            info!("Evaluating RustBERTa on SNLI");
        }
        RustBERTaOpt::Predict {
            premise,
            hypothesis,
        } => {
            predict(&premise, &hypothesis)?;
        }
    };

    Ok(())
}

fn train() -> Result<()> {
    info!("Caching pretrained transformer model");
    let model_resource_dir = cached_path_with_options(
        TRANSFORMER_MODEL,
        &cached_path::Options::default().extract(),
    )?;

    info!("Loading tokenizer and reader");
    let reader = data::Reader::new(&model_resource_dir.to_str().unwrap())?;

    info!("Reading dev data");
    let dev_data_path = cached_path::cached_path(DEV_PATH)?;
    let dev_data = reader.read(dev_data_path.to_str().unwrap())?;
    info!("Read {} instances", dev_data.len());

    info!("Reading training data");
    let train_data_path = cached_path::cached_path(TRAIN_PATH)?;
    let train_data = reader.read(train_data_path.to_str().unwrap())?;
    info!("Read {} instances", train_data.len());

    Ok(())
}

fn predict(premise: &str, hypothesis: &str) -> Result<()> {
    info!("Caching pretrained transformer model");
    let model_resource_dir = cached_path_with_options(
        TRANSFORMER_MODEL,
        &cached_path::Options::default().extract(),
    )?;

    let device = Device::cuda_if_available();
    info!("Running on {:?}", device);

    info!("Loading tokenizer");
    let tokenizer = tokenization::load_tokenizer(&model_resource_dir)?;

    info!("Loading model");
    let model = modeling::load_model(&model_resource_dir, device)?;

    info!("Tokenizing premise and hypothesis");
    let inputs = tokenizer.encode(
        premise,
        Some(hypothesis),
        MAX_SEQUENCE_LENGTH,
        &TRUNCATION_STRATEGY,
        0,
    );
    let batch = data::Batch::from_tokenized_input(&inputs, None).to_device(device);

    info!("Running forward pass");
    let logits = model.forward_on_batch(batch);
    logits.print();

    Ok(())
}

#[derive(Debug, StructOpt)]
#[structopt(
    name = "rustberta-snli",
    about = "Train or evaluate a RoBERTa SNLI model",
    setting = structopt::clap::AppSettings::ColoredHelp,
)]
enum RustBERTaOpt {
    Train,
    Evaluate,
    Predict { premise: String, hypothesis: String },
}
