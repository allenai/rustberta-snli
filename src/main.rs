use anyhow::Result;
use env_logger::Env;
use log::info;
use rust_tokenizers::{Tokenizer, TruncationStrategy};
use tch::Device;

pub mod data;
pub mod modeling;
pub mod tokenization;

use crate::modeling::TransformerSequenceClassificationModel;

const RESOURCE_ROOT: &str = "rustbert";
const MODEL: &str = "roberta";
const MAX_SEQUENCE_LENGTH: usize = 512;
const TRUNCATION_STRATEGY: TruncationStrategy = TruncationStrategy::LongestFirst;

fn main() -> Result<()> {
    env_logger::from_env(Env::default().default_filter_or("info")).init();

    let home = dirs::home_dir().unwrap();
    let model_resource_dir = home.join(RESOURCE_ROOT).join(MODEL);

    let device = Device::cuda_if_available();

    info!("Loading tokenizer");
    let tokenizer = tokenization::load_tokenizer(&model_resource_dir)?;

    info!("Loading model");
    let model = modeling::load_model(&model_resource_dir, device)?;

    let inputs = tokenizer.encode(
        "a b c",
        Some("c d e"),
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
