#!/bin/bash


CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m infer \
    --config config/cot.yaml\
    --batchsize 6\
    --num-chunks 1  \
    --chunk-idx 0  &
