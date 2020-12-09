use anyhow::Result;
use cached_path::{self, cached_path_with_options};
use env_logger::Env;
use log::info;
use structopt::StructOpt;
use tch::Device;

pub(crate) mod common;
pub mod data;
pub mod modeling;
pub mod tokenization;
pub mod training;

use data::{Instance, Reader};
use modeling::Model;
use training::Trainer;

const TRANSFORMER_MODEL: &str =
    "https://storage.googleapis.com/allennlp-public-models/rustberta.tar.gz";

const TRAIN_PATH: &str = "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_train.jsonl";
const DEV_PATH: &str = "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_dev.jsonl";
const TEST_PATH: &str = "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_test.jsonl";

fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let opt = RustBERTaOpt::from_args();

    let (reader, model) = load_components(&opt)?;

    match opt.cmd {
        RustBERTaCmd::Train => {
            info!("Training RustBERTa on SNLI");
            train(&reader, &model)?;
        }
        RustBERTaCmd::Evaluate { path } => {
            info!("Evaluating RustBERTa on {}", path);
        }
        RustBERTaCmd::Predict {
            premise,
            hypothesis,
        } => {
            predict(&reader, &model, &premise, &hypothesis)?;
        }
    };

    Ok(())
}

fn load_components(opt: &RustBERTaOpt) -> Result<(Reader, Model)> {
    info!("Caching model resources");
    let local_model_resource_dir = cached_path_with_options(
        &opt.resource_dir,
        &cached_path::Options::default().extract(),
    )?;

    info!("Loading tokenizer and reader");
    let mut reader = Reader::new(&local_model_resource_dir.to_str().unwrap())?;
    if let Some(max_instances) = opt.max_instances {
        reader.max_instances = Some(max_instances);
    }

    let device = Device::cuda_if_available();

    info!("Loading model to {:?}", device);
    let model = Model::load(&local_model_resource_dir, device)?;

    Ok((reader, model))
}

fn train(reader: &Reader, model: &Model) -> Result<()> {
    info!("Reading dev data");
    let dev_data_path = cached_path::cached_path(DEV_PATH)?;
    let dev_data = reader.read(dev_data_path.to_str().unwrap())?;
    info!("Read {} instances", dev_data.len());

    info!("Reading training data");
    let train_data_path = cached_path::cached_path(TRAIN_PATH)?;
    let train_data = reader.read(train_data_path.to_str().unwrap())?;
    info!("Read {} instances", train_data.len());

    let trainer = Trainer::builder(model, train_data)
        .validation_data(dev_data)
        .build()?;

    info!("Starting training");
    let result = trainer.train()?;
    info!("Finished training: {:?}", result);

    Ok(())
}

fn predict(reader: &Reader, model: &Model, premise: &str, hypothesis: &str) -> Result<()> {
    info!("Tokenizing premise and hypothesis");
    let batch = reader.encode_instance(&Instance::new(premise, hypothesis, None));

    info!("Running forward pass");
    let labels = model.predict(batch.to_device(model.device));
    println!("Best prediction: {}", labels[0]);

    Ok(())
}

#[derive(Debug, StructOpt)]
#[structopt(
    name = "rustberta-snli",
    about = "Train or evaluate a RoBERTa SNLI model",
    setting = structopt::clap::AppSettings::ColoredHelp,
)]
struct RustBERTaOpt {
    #[structopt(short = "m", long = "model", name = "path", default_value = TRANSFORMER_MODEL)]
    /// The path to the model resource directory.
    resource_dir: String,

    #[structopt(long = "max-instances")]
    /// The maximum number of instances to read.
    max_instances: Option<usize>,

    #[structopt(subcommand)]
    cmd: RustBERTaCmd,
}

#[derive(Debug, StructOpt)]
enum RustBERTaCmd {
    /// Train or fine-tune a new model on SNLI.
    Train,

    /// Evaluate a trained model on an SNLI dataset.
    Evaluate {
        #[structopt(short = "p", long = "path", name = "path", default_value = TEST_PATH)]
        /// The path to the data file.
        path: String,
    },

    /// Predict whether a premise and hypothesis exhibit entailment, contradiction, or neutrality.
    Predict { premise: String, hypothesis: String },
}
