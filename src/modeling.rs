use anyhow::Result;
use rust_bert::bert::BertConfig;
use rust_bert::roberta::RobertaForSequenceClassification;
use rust_bert::Config;
use std::collections::HashMap;
use std::path::Path;
use tch::{nn, no_grad, Device, Tensor};

use crate::common;
use crate::data::Batch;

pub struct Model {
    pub classifier: RobertaForSequenceClassification,
    pub device: tch::Device,
    pub vs: nn::VarStore,
}

impl Model {
    pub fn load(model_resource_dir: &Path, device: Device) -> Result<Model> {
        let config_path = model_resource_dir.join("config.json");
        let weights_path = model_resource_dir.join("model.ot");

        let id2label: HashMap<i64, String> = [
            (common::label2id("entailment") as i64, "entailment".into()),
            (
                common::label2id("contradiction") as i64,
                "contradiction".into(),
            ),
            (common::label2id("neutral") as i64, "neutral".into()),
        ]
        .iter()
        .cloned()
        .collect();
        let label2id: HashMap<String, i64> = id2label
            .iter()
            .map(|(id, label)| (label.clone(), *id))
            .collect();

        let mut config = BertConfig::from_file(&config_path);
        config.id2label = Some(id2label);
        config.label2id = Some(label2id);
        config.type_vocab_size = 2;

        let mut vs = nn::VarStore::new(device);
        let classifier = RobertaForSequenceClassification::new(&vs.root(), &config);
        vs.load_partial(weights_path)?;

        Ok(Model {
            classifier,
            device,
            vs,
        })
    }

    pub fn forward_on_batch(&self, b: Batch) -> Tensor {
        let result =
            self.classifier
                .forward_t(Some(b.token_ids), None, Some(b.type_ids), None, None, false);
        result.logits
    }

    pub fn forward_loss(&self, mut b: Batch) -> (Tensor, Tensor) {
        let labels = b
            .gold_labels
            .take()
            .expect("Batch must have gold labels to calculate loss");
        let logits = self.forward_on_batch(b);
        let loss = logits.cross_entropy_for_logits(&labels);
        let accuracy = logits.accuracy_for_logits(&labels);
        (loss, accuracy)
    }

    pub fn predict(&self, b: Batch) -> Vec<&'static str> {
        let (batch_size, _) = b.size();

        // shape: (batch_size, n_labels)
        let logits = no_grad(|| self.forward_on_batch(b));

        // shape: (batch_size,)
        let ids = logits.argmax(-1, false);

        let mut labels = Vec::with_capacity(batch_size as usize);
        for i in 0..batch_size {
            let id = ids.int64_value(&[i]);
            labels.push(common::id2label(id as u8));
        }

        labels
    }
}
