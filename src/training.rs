use crate::common;
use crate::data::Batch;
use crate::modeling::Model;
use anyhow::Result;
use rand::seq::SliceRandom;
use tch::{nn, no_grad};

pub struct Trainer<'a, O, S>
where
    O: nn::OptimizerConfig,
    S: Scheduler,
{
    // Model.
    model: &'a Model,

    // Data.
    train_data: Vec<Batch>,
    validation_data: Option<Vec<Batch>>,

    // Hyperparameters.
    epochs: u32,
    batch_size: u32,
    optimizer: nn::Optimizer<O>,
    scheduler: S,

    // Other stuff.
    rng: rand::rngs::ThreadRng,
}

impl<'a, O, S> Trainer<'a, O, S>
where
    O: nn::OptimizerConfig,
    S: Scheduler,
{
    pub fn train(mut self) -> Result<TrainResult> {
        for epoch in 0..self.epochs {
            if let Some(lr) = self.scheduler.pre_epoch_step(epoch) {
                self.optimizer.set_lr(lr);
            }

            let _ = self.train_epoch(epoch);

            if let Some(lr) = self.scheduler.post_epoch_step(epoch) {
                self.optimizer.set_lr(lr);
            }

            println!("");
        }

        // TODO: make this path configurable.
        self.model.vs.save("weights.th")?;

        // TODO:
        Ok(TrainResult {
            best_epoch: 0,
            train_loss: 0.0,
            best_validation_loss: None,
            best_validation_accuracy: None,
        })
    }

    fn train_epoch(&mut self, epoch: u32) -> EpochResult {
        self.train_data.shuffle(&mut self.rng);

        let mut train_loss_total = 0.0;
        let mut train_acc_total = 0.0;
        let num_batches = self.train_data.len() / self.batch_size as usize;
        let train_bar = common::new_epoch_bar(epoch, self.epochs, num_batches, true);

        for (batch_num, batch) in self
            .train_data
            .chunks(self.batch_size as usize)
            .map(Batch::combine)
            .enumerate()
        {
            if let Some(lr) = self.scheduler.pre_batch_step(batch_num as u32) {
                self.optimizer.set_lr(lr);
            }

            let (batch_loss, batch_acc) =
                self.model.forward_loss(batch.to_device(self.model.device));

            self.optimizer.backward_step(&batch_loss);

            let batch_loss_float = batch_loss.double_value(&[]);
            let batch_acc_float = batch_acc.double_value(&[]);
            train_loss_total += batch_loss_float;
            train_acc_total += batch_acc_float * batch.size().0 as f64;

            if let Some(lr) = self.scheduler.post_batch_step(batch_num as u32) {
                self.optimizer.set_lr(lr);
            }

            train_bar.inc(1);
            train_bar.set_message(&format!(
                "batch loss: {:.4}, batch acc: {:.4}",
                batch_loss_float / (self.batch_size as f64),
                batch_acc_float
            ));
        }

        let train_loss = train_loss_total / (self.train_data.len() as f64);
        let train_acc = train_acc_total / (self.train_data.len() as f64);
        train_bar.finish_with_message(&format!(
            "epoch train loss: {:.4}, epoch train acc: {:.4}",
            train_loss, train_acc
        ));

        if self.validation_data.is_none() {
            return EpochResult {
                train_loss,
                train_acc,
                validation_loss: None,
                validation_acc: None,
            };
        }

        let validation_data = self.validation_data.as_ref().unwrap();
        let mut validation_loss_total = 0.0;
        let mut validation_acc_total = 0.0;
        let num_batches = validation_data.len() / self.batch_size as usize;
        let validation_bar = common::new_epoch_bar(epoch, self.epochs, num_batches, false);

        no_grad(|| {
            for batch in validation_data
                .chunks(self.batch_size as usize)
                .map(Batch::combine)
            {
                let (batch_loss, batch_acc) =
                    self.model.forward_loss(batch.to_device(self.model.device));

                let batch_loss_float = batch_loss.double_value(&[]);
                let batch_acc_float = batch_acc.double_value(&[]);
                validation_loss_total += batch_loss_float;
                validation_acc_total += batch_acc_float * batch.size().0 as f64;

                validation_bar.inc(1);
                validation_bar.set_message(&format!(
                    "batch loss: {:.4}, batch acc: {:.4}",
                    batch_loss_float / (self.batch_size as f64),
                    batch_acc_float
                ));
            }
        });

        let validation_loss = validation_loss_total / (validation_data.len() as f64);
        let validation_acc = validation_acc_total / (validation_data.len() as f64);
        validation_bar.finish_with_message(&format!(
            "epoch valid loss: {:.4}, epoch valid acc: {:.4}",
            validation_loss, validation_acc
        ));

        EpochResult {
            train_loss,
            train_acc,
            validation_loss: Some(validation_loss),
            validation_acc: Some(validation_acc),
        }
    }
}

impl<'a> Trainer<'a, nn::AdamW, LinearSchedulerWithWarmup> {
    pub fn builder(model: &'a Model, train_data: Vec<Batch>) -> TrainerBuilder<nn::AdamW> {
        TrainerBuilder::new(model, train_data)
    }
}

struct TrainerConfig<'a> {
    model: &'a Model,
    train_data: Vec<Batch>,
    validation_data: Option<Vec<Batch>>,
    batch_size: u32,
    epochs: u32,
    lr: f64,
    warmup_steps: u32,
}

pub struct TrainerBuilder<'a, O>
where
    O: nn::OptimizerConfig,
{
    config: TrainerConfig<'a>,
    // We keep `optimizer_config` out of `config` so that we can use `..self.config` syntax
    // in the `TrainerBuilder::optimizer()` method.
    optimizer_config: O,
}

impl<'a> TrainerBuilder<'a, nn::AdamW> {
    pub fn new(model: &'a Model, train_data: Vec<Batch>) -> TrainerBuilder<nn::AdamW> {
        Self {
            config: TrainerConfig {
                model,
                train_data,
                validation_data: None,
                batch_size: 32,
                epochs: 2,
                lr: 2e-5,
                warmup_steps: 1000,
            },
            optimizer_config: nn::adamw(0.9, 0.999, 0.1),
        }
    }
}

impl<'a, O> TrainerBuilder<'a, O>
where
    O: nn::OptimizerConfig,
{
    pub fn build(self) -> Result<Trainer<'a, O, LinearSchedulerWithWarmup>> {
        let optimizer = self
            .optimizer_config
            .build(&self.config.model.vs, self.config.lr)?;

        let scheduler = LinearSchedulerWithWarmup {
            warmup_steps: self.config.warmup_steps,
            total_steps: (self.config.train_data.len() as u32 / self.config.batch_size)
                * self.config.epochs,
            steps: 0,
            base_lr: self.config.lr,
            end_lr: 0.0,
        };

        Ok(Trainer {
            model: self.config.model,
            train_data: self.config.train_data,
            validation_data: self.config.validation_data,
            batch_size: self.config.batch_size,
            epochs: self.config.epochs,
            optimizer,
            scheduler,
            rng: rand::thread_rng(),
        })
    }

    pub fn validation_data(mut self, validation_data: Vec<Batch>) -> Self {
        self.config.validation_data = Some(validation_data);
        self
    }

    pub fn lr(mut self, lr: f64) -> Self {
        self.config.lr = lr;
        self
    }

    pub fn batch_size(mut self, batch_size: u32) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    pub fn epochs(mut self, epochs: u32) -> Self {
        self.config.epochs = epochs;
        self
    }

    pub fn warmup_steps(mut self, warmup_steps: u32) -> Self {
        self.config.warmup_steps = warmup_steps;
        self
    }

    pub fn optimizer<U>(self, optimizer_config: U) -> TrainerBuilder<'a, U>
    where
        U: nn::OptimizerConfig,
    {
        TrainerBuilder {
            config: self.config,
            optimizer_config,
        }
    }
}

pub trait Scheduler {
    fn pre_batch_step(&mut self, _batch_num: u32) -> Option<f64> {
        None
    }

    fn post_batch_step(&mut self, _batch_num: u32) -> Option<f64> {
        None
    }

    fn pre_epoch_step(&mut self, _epoch: u32) -> Option<f64> {
        None
    }

    fn post_epoch_step(&mut self, _epoch: u32) -> Option<f64> {
        None
    }
}

pub struct LinearSchedulerWithWarmup {
    warmup_steps: u32,
    total_steps: u32,
    steps: u32,
    base_lr: f64,
    end_lr: f64,
}

impl Scheduler for LinearSchedulerWithWarmup {
    fn pre_batch_step(&mut self, _batch_num: u32) -> Option<f64> {
        self.steps += 1;
        let lr = if self.steps < self.warmup_steps {
            self.base_lr * (self.steps as f64 / self.warmup_steps as f64)
        } else if self.steps > self.total_steps {
            self.end_lr
        } else {
            let current_decay_steps = self.total_steps - self.steps;
            let total_decay_steps = self.total_steps - self.warmup_steps;
            let factor = current_decay_steps as f64 / total_decay_steps as f64;
            factor * (self.base_lr - self.end_lr) + self.end_lr
        };
        Some(lr)
    }
}

#[derive(Debug)]
pub struct TrainResult {
    pub best_epoch: u32,
    pub train_loss: f64,
    pub best_validation_loss: Option<f64>,
    pub best_validation_accuracy: Option<f64>,
}

#[derive(Debug)]
struct EpochResult {
    train_loss: f64,
    train_acc: f64,
    validation_loss: Option<f64>,
    validation_acc: Option<f64>,
}
