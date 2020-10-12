use anyhow::Result;
use rust_tokenizers::RobertaTokenizer;
use std::path::Path;

pub fn load_tokenizer<P: AsRef<Path>>(model_resource_dir: P) -> Result<RobertaTokenizer> {
    let vocab_path = model_resource_dir.as_ref().join("vocab.txt");
    let merges_path = model_resource_dir.as_ref().join("merges.txt");

    Ok(RobertaTokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        false, // lowercase
        true,  // add prefix space
    )?)
}
