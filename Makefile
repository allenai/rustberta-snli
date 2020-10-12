.PHONY : build
build :
	cargo build

.PHONY : format
format :
	cargo fmt --

.PHONY : lint
lint :
	cargo clippy --all-targets --all-features -- -D warnings
