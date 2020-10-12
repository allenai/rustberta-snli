use anyhow::Result;
use cached_path::{self, cached_path_with_options};
use env_logger::Env;
use log::info;
use rust_tokenizers::{Tokenizer, TruncationStrategy};
use structopt::StructOpt;
use tch::Device;

pub mod data;
pub mod modeling;
pub mod tokenization;

use crate::modeling::TransformerSequenceClassificationModel;

const TRANSFORMER_MODEL: &str =
    "https://storage.googleapis.com/allennlp-public-models/rustberta.tar.gz";
const MAX_SEQUENCE_LENGTH: usize = 512;
const TRUNCATION_STRATEGY: TruncationStrategy = TruncationStrategy::LongestFirst;

fn main() -> Result<()> {
    env_logger::from_env(Env::default().default_filter_or("info")).init();
    let opt = RustBERTaOpt::from_args();

    match opt {
        RustBERTaOpt::Train => {
            info!("Training RustBERTa on SNLI");
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

fn predict(premise: &str, hypothesis: &str) -> Result<()> {
    info!("Caching pretrained transformer model");
    let model_resource_dir = cached_path_with_options(
        TRANSFORMER_MODEL,
        &cached_path::Options::default().extract(),
    )?;

    let device = Device::cuda_if_available();

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
    let batch = data::Batch::from_tokenized_input(&inputs);

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
