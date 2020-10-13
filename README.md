# rustberta-snli (WORK IN PROGRESS)

A Rust implementation of a RoBERTa classification model for the SNLI dataset.

## Setup

First of all, you'll need the Rust toolchain installed. If you're already a Rustacean, skip ahead to the "Additional setup for CUDA" section.

Now, luckily, installing Rust is nothing like installing a proper Python environment, i.e. it doesn't require a PhD in system administration or
the courage to blindly run every sudo command you can find on Stack Overflow until something works or completely breaks your computer.

All you have to do is run this:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then just make sure `~/.cargo/bin` is in your `$PATH`, and you're good to go.
You can test for success by running `rustup --version` and `cargo --version`.

> `rustup` can be used to update your toolchain when a new version of Rust is released (which happens monthly). `cargo` is used to compile, run, and test your code, as well as to build documentation, publish your crate (the Rust term for a module/library) to [crates.io](crates.io), and install binaries from other crates on [crates.io](crates.io).

## Additional setup for CUDA

If you have CUDA-enabled GPUs available on your machine, you'll probably want to compile
this library with CUDA support.

To do that, you just need to download the right version of LibTorch from the PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

Then unzip the downloaded file to someplace safe like `~/torch/libtorch` and set the environment variables `LIBTORCH=$HOME/torch/libtorch` (or wherever you unzipped it) and
`LD_LIBRARY_PATH=$HOME/torch/libtorch/lib:$LD_LIBRARY_PATH`.

## Compiling and running

To see all of the available commands, run

```bash
cargo run -- --help
```

For example, to get a prediction, run

```
cargo run -- predict 'A man inspects the uniform of a figure in some East Asian country' 'The man is sleeping'
```

> If you compiled with CUDA support, you should see `"INFO: Running on Cuda(0)"` or something like that in the logs.
