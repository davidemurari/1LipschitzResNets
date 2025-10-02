#!/usr/bin/env bash

for layers in 5 10 20; do
    for inner_dim in 50 100 200; do
        python mainMNIST.py \
            --layers $layers \
            --inner_dim $inner_dim \
            --epochs 100 \
            --theorem4
    done
done