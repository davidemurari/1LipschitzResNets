#!/usr/bin/env bash

for layers in 2 4 8 16 32 64; do
    python main.py \
        --layers $layers \
        --epochs 1000 
done

for layers in 2 4 8 16 32 64; do
    python main.py \
        --theorem4 \
        --layers $layers \
        --epochs 1000 
done

for inner_dim in 10 20 30 40; do
    python main.py \
        --layers 10 \
        --inner_dim $inner_dim \
        --epochs 1000 
done

for inner_dim in 10 20 30 40; do
    python main.py \
        --theorem4 \
        --layers 10 \
        --inner_dim $inner_dim \
        --epochs 1000 
done