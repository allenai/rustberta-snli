pub(crate) fn label2id(label: &str) -> u8 {
    match label {
        "entailment" => 0,
        "contradiction" => 1,
        "neutral" => 2,
        _ => panic!("Bad label {}", label),
    }
}

pub(crate) fn new_progress_bar() -> indicatif::ProgressBar {
    let progress_bar = indicatif::ProgressBar::new_spinner().with_style(
        indicatif::ProgressStyle::default_spinner()
            .template("{pos} [{per_sec}, {elapsed}] {spinner}"),
    );
    progress_bar.set_draw_delta(1_117);
    progress_bar
}
