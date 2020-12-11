# RustBERTa-SNLI

A Rust implementation of a RoBERTa classification model for the [SNLI dataset](https://nlp.stanford.edu/projects/snli/), with support for fine-tuning, predicting, and serving.
This is built on top of [tch-rs](https://github.com/LaurentMazare/tch-rs) and [rust-bert](https://github.com/guillaume-be/rust-bert).

## Background

This was the result of the **AI2 2020 Employee Hackathon**.
The motivation for this project was to demonstrate that Rust is already a viable alternative to Python
for deep learning projects in the context of both research and production.

## Setup

RustBERTa-SNLI is packaged as a Rust binary, and will work on any operating system that [PyTorch](https://pytorch.org/) supports.

The only prerequisite is that you have the Rust toolchain installed.
So if you're already a Rustacean, skip ahead to the ["Additional setup for CUDA"](#additional-setup-for-cuda) section (optional).

Now, luckily, installing Rust is nothing like installing a proper Python environment, i.e. it doesn't require a PhD in system administration or
the courage to blindly run every sudo command you can find on Stack Overflow until something works or completely breaks your computer.

All you have to do is run this:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then just make sure `~/.cargo/bin` is in your `$PATH`, and you're good to go.
You can test for success by running `rustup --version` and `cargo --version`.

> `rustup` can be used to update your toolchain when a new version of Rust is released (which happens monthly). `cargo` is used to compile, run, and test your code, as well as to build documentation, publish your crate (the Rust term for a module/library) to [crates.io](crates.io), and install binaries from other crates on [crates.io](crates.io).

### Additional setup for CUDA

If you have CUDA-enabled GPUs available on your machine, you'll probably want to compile
this library with CUDA support.

To do that, you just need to download the right version of LibTorch from the PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

Then unzip the downloaded file to someplace safe like `~/torch/libtorch` and set the environment variables:

```bash
LIBTORCH="$HOME/torch/libtorch"  # or wherever you unzipped it
LD_LIBRARY_PATH="$HOME/torch/libtorch/lib:$LD_LIBRARY_PATH"
```

‼️ NOTE: it's important that you do this step *before* trying to compile the Rust binary with `cargo`. But if you accidentally start the build
for this step, just delete the `target/` directory and start over.

## Compiling and running

To build the release binary, just run `make`.

To see all of the available commands, run

```bash
./roberta-snli --help
```

For example, to fine-tune a pretrained RoBERTa model, run

```bash
./roberta-snli train --out weights.ot
```

To interactively get predictions with a fine-tuned model, run

```bash
./roberta-snli predict --weigths weights.ot
```

To evaluate a fine-tuned model on the test set, run

```bash
./roberta-snli evaluate
```

And to serve a fine-tuned model as a production-grade webservice with batched prediction, run

```bash
./roberta-snli serve
```

This will serve on port 3030 by default. You can then test it out by running:

```bash
curl \
    -d '{"premise":"A soccer game with multiple males playing.","hypothesis":"Some men are playing a sport."}' \
    -H "Content-Type: application/json" \
    http://localhost:3030/predict
```

You can also test the batching functionality by sending a bunch of requests at once with:

```bash
./scripts/test_server.sh
```
