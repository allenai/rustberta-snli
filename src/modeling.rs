use anyhow::Result;
use rust_bert::bert::BertConfig;
use rust_bert::roberta::RobertaForSequenceClassification;
use rust_bert::Config;
use std::collections::HashMap;
use std::path::Path;
use tch::{nn, Device, Tensor};

use crate::data::Batch;

pub fn load_model(
    model_resource_dir: &Path,
    device: Device,
) -> Result<impl TransformerSequenceClassificationModel> {
    let config_path = model_resource_dir.join("config.json");
    let weights_path = model_resource_dir.join("model.ot");

    let id2label: HashMap<i64, String> = [
        (0, "entailment".into()),
        (1, "contradiction".into()),
        (2, "neutral".into()),
    ]
    .iter()
    .cloned()
    .collect();
    let label2id: HashMap<String, i64> = id2label
        .iter()
        .map(|(id, label)| (label.clone(), id.clone()))
        .collect();

    let mut config = BertConfig::from_file(&config_path);
    config.id2label = Some(id2label);
    config.label2id = Some(label2id);
    config.type_vocab_size = 2;

    let mut vs = nn::VarStore::new(device);
    let model = RobertaForSequenceClassification::new(&vs.root(), &config);
    vs.load_partial(weights_path)?;

    Ok(model)
}

pub trait TransformerSequenceClassificationModel {
    /// Returns logits.
    fn forward_on_batch(&self, b: Batch) -> Tensor;
}

impl TransformerSequenceClassificationModel for RobertaForSequenceClassification {
    fn forward_on_batch(&self, b: Batch) -> Tensor {
        let result = self.forward_t(Some(b.token_ids), None, Some(b.type_ids), None, None, false);
        result.logits
    }
}
