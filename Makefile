roberta-snli : .FORCE
	cargo build --release

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

.PHONY : post
post :
	curl \
		-d '{"premise":"A soccer game with multiple males playing.","hypothesis":"Some men are playing a sport."}' \
		-H "Content-Type: application/json" \
		http://localhost:3030/predict

.FORCE :
