use anyhow::Result;
use rust_tokenizers::tokenizer::RobertaTokenizer;
use std::path::Path;

pub const PAD_TOKEN_ID: i64 = 1;

pub fn load_tokenizer<P: AsRef<Path>>(vocab_path: P, merges_path: P) -> Result<RobertaTokenizer> {
    Ok(RobertaTokenizer::from_file(
        vocab_path.as_ref().to_str().unwrap(),
        merges_path.as_ref().to_str().unwrap(),
        false, // lowercase
        true,  // add prefix space
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_tokenizers::tokenizer::Tokenizer;
    use rust_tokenizers::vocab::{RobertaVocab, Vocab};

    #[test]
    fn test_tokenizer() {
        let tokenizer = load_tokenizer(
            "test_fixtures/tokenizer/vocab.txt",
            "test_fixtures/tokenizer/merges.txt",
        )
        .unwrap();

        // Make sure PAD_TOKEN_ID is correct.
        let pad_token = RobertaVocab::pad_value();
        let pad_token_id = tokenizer.vocab().special_values().get(pad_token).unwrap();
        assert_eq!(*pad_token_id, PAD_TOKEN_ID);
    }
}
