.PHONY : build
build :
	cargo build

.PHONY : test
test :
	cargo test --all-features

.PHONY : format
format :
	cargo fmt --

.PHONY : lint
lint :
	cargo clippy --all-targets --all-features -- -D warnings
