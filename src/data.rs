use rust_tokenizers::TokenizedInput;
use tch::Tensor;

pub struct Batch {
    pub token_ids: Tensor,
    pub type_ids: Tensor,
}

impl Batch {
    pub fn new(token_ids: &[i64], type_ids: &[i8]) -> Self {
        Self {
            token_ids: Tensor::of_slice(token_ids).unsqueeze(0),
            type_ids: Tensor::of_slice(type_ids)
                .totype(tch::Kind::Int64)
                .unsqueeze(0),
        }
    }

    pub fn from_tokenized_input(input: &TokenizedInput) -> Self {
        Self::new(&input.token_ids, &input.segment_ids)
    }
}
