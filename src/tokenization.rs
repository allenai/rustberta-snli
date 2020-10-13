use anyhow::Result;
use rust_tokenizers::RobertaTokenizer;
use std::path::Path;

pub const PAD_TOKEN_ID: i64 = 1;

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

#[cfg(test)]
mod tests {
    use super::*;
    use rust_tokenizers::Tokenizer;

    #[test]
    fn test_tokenizer() {
        let tokenizer = load_tokenizer("test_fixtures/tokenizer").unwrap();

        // Make sure PAD_TOKEN_ID is correct.
        let tokenized_pad = tokenizer.convert_tokens_to_ids(&tokenizer.tokenize("<pad>"));
        assert_eq!(tokenized_pad, vec![PAD_TOKEN_ID]);
    }
}
