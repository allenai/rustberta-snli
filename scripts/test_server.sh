#!/bin/bash

set -e

for i in {1..16}; do
    curl \
        -d '{"premise":"A soccer game with multiple males playing.","hypothesis":"Some men are playing a sport."}' \
        -H "Content-Type: application/json" \
        http://localhost:3030/predict > /dev/null 2> /dev/null &
done
