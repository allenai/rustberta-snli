use anyhow::Result;
use cached_path::{self, cached_path_with_options};
use env_logger::Env;
use log::{info, warn};
use std::io::{stdin, stdout, Write};
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

const PRETRAINED_MODEL: &str =
    "https://storage.googleapis.com/allennlp-public-models/rustberta.tar.gz";
const FINE_TUNED_MODEL: &str =
    "https://storage.googleapis.com/allennlp-public-models/rustberta-snli.ot";
const TRAIN_PATH: &str = "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_train.jsonl";
const DEV_PATH: &str = "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_dev.jsonl";
// const TEST_PATH: &str = "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_test.jsonl";

fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let opt = RustBERTaOpt::from_args();

    let (reader, model) = load_components(&opt)?;

    match opt.cmd {
        RustBERTaCmd::Train(train_opts) => {
            info!("Training RustBERTa on SNLI");
            train(reader, model, &train_opts)?;
        }
        RustBERTaCmd::Predict => {
            predict(reader, model)?;
        }
    };

    Ok(())
}

fn load_components(opt: &RustBERTaOpt) -> Result<(Reader, Model)> {
    let weights_path = match &opt.weights {
        Some(weights_path) => cached_path::cached_path(weights_path)?,
        None => match opt.cmd {
            RustBERTaCmd::Train(_) => {
                info!("Caching pretrained model");
                let pretrained_model_dir = cached_path_with_options(
                    PRETRAINED_MODEL,
                    &cached_path::Options::default().extract(),
                )?;
                pretrained_model_dir.join("model.ot")
            }
            _ => {
                info!("Caching fine-tuned model");
                cached_path::cached_path(FINE_TUNED_MODEL)?
            }
        },
    };
    let vocab_path = cached_path::cached_path(&opt.vocab)?;
    let merges_path = cached_path::cached_path(&opt.merges)?;
    let config_path = cached_path::cached_path(&opt.config)?;

    info!("Loading tokenizer and reader");
    let reader = Reader::new(&vocab_path, &merges_path)?;

    let device = Device::cuda_if_available();

    info!("Loading model to {:?}", device);
    let model = Model::load(&config_path, &weights_path, device)?;

    Ok((reader, model))
}

fn train(mut reader: Reader, model: Model, opt: &TrainOpts) -> Result<()> {
    reader.max_instances = opt.max_instances;

    info!("Reading dev data");
    let dev_data_path = cached_path::cached_path(DEV_PATH)?;
    let dev_data = reader.read(dev_data_path.to_str().unwrap())?;
    info!("Read {} instances", dev_data.len());

    info!("Reading training data");
    let train_data_path = cached_path::cached_path(TRAIN_PATH)?;
    let train_data = reader.read(train_data_path.to_str().unwrap())?;
    info!("Read {} instances", train_data.len());

    let trainer = Trainer::builder(&model, train_data)
        .lr(opt.lr)
        .warmup_steps(opt.warmup_steps)
        .epochs(opt.epochs)
        .batch_size(opt.batch_size)
        .validation_data(dev_data)
        .build()?;

    info!("Starting training");
    let result = trainer.train()?;
    info!("Finished training: {:?}", result);

    Ok(())
}

fn predict(reader: Reader, model: Model) -> Result<()> {
    println!("Starting interactive session, press 'q' or CTRL-C at any time to quit.");

    loop {
        print!("Enter a premise: ");
        let mut premise = String::new();
        let _ = stdout().flush();
        stdin().read_line(&mut premise)?;
        premise = premise.trim().into();
        if premise == "" {
            warn!("You must enter something for the premise");
            continue;
        } else if premise == "q" {
            println!("👋 See ya!");
            break;
        }

        print!("Enter a hypothesis: ");
        let mut hypothesis = String::new();
        let _ = stdout().flush();
        stdin().read_line(&mut hypothesis)?;
        hypothesis = hypothesis.trim().into();
        if hypothesis == "" {
            warn!("You must enter something for the hypothesis");
            continue;
        } else if hypothesis == "q" {
            println!("👋 See ya!");
            break;
        }

        let batch = reader.encode_instance(&Instance::new(&premise, &hypothesis, None));
        let labels = model.predict(batch.to_device(model.device));
        let label = labels[0];
        if label == "entailment" {
            println!("✅ {}\n", label);
        } else if label == "contradiction" {
            println!("❌ {}\n", label);
        } else {
            println!("❓ {}\n", label);
        }
    }

    Ok(())
}

#[derive(Debug, StructOpt)]
#[structopt(
    name = "rustberta-snli",
    about = "Train or evaluate a RoBERTa SNLI model",
    setting = structopt::clap::AppSettings::ColoredHelp,
)]
struct RustBERTaOpt {
    #[structopt(long = "config", default_value = "config/config.json")]
    /// The path (local or remote) to the RustBERT config file.
    config: String,

    #[structopt(long = "vocab", default_value = "config/vocab.txt")]
    /// The path (local or remote) to the vocab file.
    vocab: String,

    #[structopt(long = "merges", default_value = "config/merges.txt")]
    /// The path (local or remote) to the merges file.
    merges: String,

    #[structopt(long = "weights")]
    /// The path (local or remote) to the serialized variable store.
    weights: Option<String>,

    #[structopt(subcommand)]
    cmd: RustBERTaCmd,
}

#[derive(Debug, StructOpt)]
enum RustBERTaCmd {
    #[structopt(setting = structopt::clap::AppSettings::ColoredHelp)]
    /// Train or fine-tune a new model on SNLI.
    Train(TrainOpts),

    #[structopt(setting = structopt::clap::AppSettings::ColoredHelp)]
    /// (Interactive) Predict whether a premise and hypothesis exhibit entailment, contradiction, or neutrality.
    Predict,
}

#[derive(Debug, StructOpt)]
struct TrainOpts {
    #[structopt(long = "lr", default_value = "2e-5")]
    /// The learning rate.
    lr: f64,

    #[structopt(long = "warmup-steps", default_value = "1000")]
    /// The number of warmup steps for the learning rate scheduler.
    warmup_steps: u32,

    #[structopt(long = "batch-size", default_value = "32")]
    /// The learning rate.
    batch_size: u32,

    #[structopt(long = "epochs", default_value = "2")]
    /// The number of epochs to train for.
    epochs: u32,

    #[structopt(long = "max-instances")]
    /// The maximum number of instances to read.
    max_instances: Option<usize>,
}
