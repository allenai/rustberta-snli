pub(crate) fn label2id(label: &str) -> u8 {
    match label {
        "entailment" => 0,
        "contradiction" => 1,
        "neutral" => 2,
        _ => panic!("Bad label {}", label),
    }
}

pub(crate) fn id2label(id: u8) -> &'static str {
    match id {
        0 => "entailment",
        1 => "contradiction",
        2 => "neutral",
        _ => panic!("Bad label ID {}", id),
    }
}

pub(crate) fn new_spinner() -> indicatif::ProgressBar {
    let progress_bar = indicatif::ProgressBar::new_spinner().with_style(
        indicatif::ProgressStyle::default_spinner()
            .template("{pos} [{per_sec}, {elapsed}] {spinner}"),
    );
    progress_bar.set_draw_delta(1_117);
    progress_bar
}

pub(crate) enum BarType {
    Train,
    Validation,
    Test,
}

pub(crate) fn new_epoch_bar(
    epoch: u32,
    total_epochs: u32,
    steps: usize,
    bar_type: BarType,
) -> indicatif::ProgressBar {
    let template = match bar_type {
        BarType::Train =>      "Training   [{elapsed_precise} < {eta_precise}] [{bar:40.green}] {pos:>5}/{len:5} ({percent:>3}%), {msg}",
        BarType::Validation => "Validating [{elapsed_precise} < {eta_precise}] [{bar:40.cyan/blue}] {pos:>5}/{len:5} ({percent:>3}%), {msg}",
        BarType::Test =>       "Evaluating [{elapsed_precise} < {eta_precise}] [{bar:40.cyan/blue}] {pos:>5}/{len:5} ({percent:>3}%), {msg}",
    };
    indicatif::ProgressBar::new(steps as u64).with_style(
        indicatif::ProgressStyle::default_bar()
            .template(&format!("{}/{} {}", epoch + 1, total_epochs, template))
            .progress_chars("=> "),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_and_label_ids() {
        for id in 0..3 {
            let label = id2label(id);
            assert_eq!(label2id(label), id);
        }
    }
}
