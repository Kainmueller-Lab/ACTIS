#!/bin/bash
unset CUDA_VISIBLE_DEVICES
python src/train_supervised.py --params configs/params_supervised.toml