use anyhow::Result;
use batched_fn::batched_fn;
use cached_path::{self, cached_path_with_options};
use env_logger::Env;
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::io::{stdin, stdout, Write};
use structopt::StructOpt;
use tch::{Cuda, Device};
use warp::{http::StatusCode, reject::Reject, Filter, Rejection};

pub(crate) mod common;
pub mod data;
pub mod modeling;
pub mod tokenization;
pub mod training;

use data::{Batch, Instance, Reader};
use modeling::Model;
use training::Trainer;

const PRETRAINED_MODEL: &str =
    "https://storage.googleapis.com/allennlp-public-models/rustberta.tar.gz";
const FINE_TUNED_MODEL: &str =
    "https://storage.googleapis.com/allennlp-public-models/rustberta-snli.ot";
const DEFAULT_CONFIG: &str = "config/config.json";
const DEFAULT_VOCAB: &str = "config/vocab.txt";
const DEFAULT_MERGES: &str = "config/merges.txt";
const TRAIN_PATH: &str = "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_train.jsonl";
const DEV_PATH: &str = "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_dev.jsonl";
// const TEST_PATH: &str = "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_test.jsonl";

fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let opt = RustBERTaOpt::from_args();

    match opt.cmd {
        RustBERTaCmd::Train(train_opts) => {
            train(&train_opts)?;
        }
        RustBERTaCmd::Predict(predict_opts) => {
            predict(&predict_opts)?;
        }
        RustBERTaCmd::Serve => {
            let mut rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async { serve().await })?;
        }
    };

    Ok(())
}

fn train(opt: &TrainOpts) -> Result<()> {
    let weights_path = match &opt.weights {
        Some(weights_path) => cached_path::cached_path(weights_path)?,
        None => {
            info!("Caching pretrained model");
            let pretrained_model_dir = cached_path_with_options(
                PRETRAINED_MODEL,
                &cached_path::Options::default().extract(),
            )?;
            pretrained_model_dir.join("model.ot")
        }
    };
    let vocab_path = cached_path::cached_path(&opt.vocab)?;
    let merges_path = cached_path::cached_path(&opt.merges)?;
    let config_path = cached_path::cached_path(&opt.config)?;

    info!("Loading tokenizer and reader");
    let mut reader = Reader::new(&vocab_path, &merges_path)?;
    reader.max_instances = opt.max_instances;

    let device = Device::cuda_if_available();
    info!("Loading model to {:?}", device);
    let model = Model::load(&config_path, &weights_path, device)?;
    info!("Training RustBERTa on SNLI");

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
        .out_path(&opt.out)
        .epochs(opt.epochs)
        .batch_size(opt.batch_size)
        .validation_data(dev_data)
        .build()?;

    info!("Starting training");
    let result = trainer.train()?;
    info!("Successfully finished training");
    if let (Some(best_epoch), Some(best_val_acc), Some(best_val_loss)) = (
        result.best_epoch,
        result.best_validation_accuracy,
        result.best_validation_loss,
    ) {
        info!("Best epoch: {}", best_epoch);
        info!("Best epoch train loss: {:.4}", result.train_loss);
        info!("Best validation loss:  {:.4}", best_val_loss);
        info!("Best validation acc:   {:.4}", best_val_acc);
    } else {
        info!("Training loss: {:.4}", result.train_loss);
    }

    Ok(())
}

fn predict(opt: &PredictOpts) -> Result<()> {
    let weights_path = match &opt.weights {
        Some(weights_path) => cached_path::cached_path(weights_path)?,
        None => {
            info!("Caching fine-tuned model");
            cached_path::cached_path(FINE_TUNED_MODEL)?
        }
    };
    let vocab_path = cached_path::cached_path(&opt.vocab)?;
    let merges_path = cached_path::cached_path(&opt.merges)?;
    let config_path = cached_path::cached_path(&opt.config)?;

    info!("Loading tokenizer and reader");
    let reader = Reader::new(&vocab_path, &merges_path)?;

    let device = Device::cuda_if_available();

    info!("Loading model to {:?}", device);
    let model = Model::load(&config_path, &weights_path, device)?;

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

async fn serve() -> Result<()> {
    // POST /predict/ {"premise":"...","hypothesis":""}
    let routes = warp::post()
        .and(warp::path("predict"))
        .and(warp::body::content_length_limit(1024 * 16))
        .and(warp::body::json())
        .and_then(batched_predict)
        .recover(handle_rejection);

    warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;
    Ok(())
}

async fn batched_predict(input: Input) -> Result<impl warp::Reply, Rejection> {
    info!("Received input");

    // Using the `batched_fn` macro, we run the model in a separate thread and
    // batch input together to make better use of the GPU.
    //
    // NOTE: this is only more efficient if you have a GPU. If serving the model
    // on CPU this just adds overhead.
    let batched_predict = batched_fn! {
        handler = |batch: Vec<Input>, reader: &Reader, model: &Model| -> Vec<&'static str> {
            info!("Running batch of size {}", batch.len());
            let instances: Vec<Batch> = batch.into_iter().map(|input| reader.encode_instance(
                    &Instance::new(&input.premise, &input.hypothesis, None)
            )).collect();
            let batch = Batch::combine(&instances).to_device(model.device);
            model.predict(batch)
        };
        config = {
            max_batch_size: if Cuda::cudnn_is_available() { 8 } else { 1 },
            max_delay: 100,
            channel_cap: Some(20),
        };
        context = {
            reader: {
                Reader::new(DEFAULT_VOCAB, DEFAULT_MERGES).expect("Failed to load reader")
            },
            model: {
                let device = Device::cuda_if_available();
                let weights_path = cached_path::cached_path(FINE_TUNED_MODEL).expect("Failed to download fine-tuned model");
                Model::load(DEFAULT_CONFIG, weights_path.to_str().unwrap(), device).expect("Failed to load model")
            },
        };
    };

    batched_predict(input).await.map_err(|e| match e {
        batched_fn::Error::Full => {
            error!("At capacity!");
            warp::reject::custom(CapacityFullError)
        }
        _ => {
            // This should only happen if the handler thread crashed.
            panic!("{:?}", e);
        }
    })
}

#[derive(Debug, Deserialize, Serialize)]
struct Input {
    premise: String,
    hypothesis: String,
}

#[derive(Debug)]
struct CapacityFullError;

impl Reject for CapacityFullError {}

async fn handle_rejection(err: Rejection) -> Result<impl warp::Reply, Infallible> {
    let code;
    let message;

    if err.is_not_found() {
        code = StatusCode::NOT_FOUND;
        message = "NOT_FOUND";
    } else if let Some(CapacityFullError) = err.find() {
        code = StatusCode::SERVICE_UNAVAILABLE;
        message = "AT_CAPACITY";
    } else if err.find::<warp::reject::MethodNotAllowed>().is_some() {
        // We can handle a specific error, here METHOD_NOT_ALLOWED,
        // and render it however we want
        code = StatusCode::METHOD_NOT_ALLOWED;
        message = "METHOD_NOT_ALLOWED";
    } else {
        // We should have expected this... Just log and say its a 500
        error!("unhandled rejection: {:?}", err);
        code = StatusCode::INTERNAL_SERVER_ERROR;
        message = "UNHANDLED_REJECTION";
    }

    Ok(warp::reply::with_status(message, code))
}

#[derive(Debug, StructOpt)]
#[structopt(
    name = "rustberta-snli",
    about = "Train, evaluate, or serve a RoBERTa SNLI model",
    setting = structopt::clap::AppSettings::ColoredHelp,
)]
struct RustBERTaOpt {
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
    Predict(PredictOpts),

    #[structopt(setting = structopt::clap::AppSettings::ColoredHelp)]
    /// Serve a model as a production-grade webservice with batched prediction.
    Serve,
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

    #[structopt(long = "config", default_value = DEFAULT_CONFIG)]
    /// The path (local or remote) to the RustBERT config file.
    config: String,

    #[structopt(long = "vocab", default_value = DEFAULT_VOCAB)]
    /// The path (local or remote) to the vocab file.
    vocab: String,

    #[structopt(long = "merges", default_value = DEFAULT_MERGES)]
    /// The path (local or remote) to the merges file.
    merges: String,

    #[structopt(long = "weights")]
    /// The path (local or remote) to the initial variable store.
    weights: Option<String>,

    #[structopt(long = "out", default_value = "weights.ot")]
    /// The filename to save the final weights variable store to.
    out: String,
}

#[derive(Debug, StructOpt)]
struct PredictOpts {
    #[structopt(long = "config", default_value = DEFAULT_CONFIG)]
    /// The path (local or remote) to the RustBERT config file.
    config: String,

    #[structopt(long = "vocab", default_value = DEFAULT_VOCAB)]
    /// The path (local or remote) to the vocab file.
    vocab: String,

    #[structopt(long = "merges", default_value = DEFAULT_MERGES)]
    /// The path (local or remote) to the merges file.
    merges: String,

    #[structopt(long = "weights")]
    /// The path (local or remote) to the serialized variable store.
    weights: Option<String>,
}
