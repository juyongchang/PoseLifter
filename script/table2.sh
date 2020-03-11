#!/bin/bash

# script for table 1

# baseline
CUDA_VISIBLE_DEVICES=2 python ./src/main_lift.py \
    -dataset_test 'h36m' -dataset_train 'h36m' -protocol 1 -noise 0 -canonical \
    -network 'resnet' -mode 0 -num_layers 2 -num_features 4096 \
    -opt_method rmsprop -lr 1e-3 -num_epochs 300 -batch_size 64

# + location
CUDA_VISIBLE_DEVICES=2 python ./src/main_lift.py \
    -dataset_test 'h36m' -dataset_train 'h36m' -protocol 1 -noise 0 -canonical \
    -network 'resnet' -mode 3 -num_layers 2 -num_features 4096 \
    -opt_method rmsprop -lr 1e-3 -num_epochs 300 -batch_size 64

# + scale
CUDA_VISIBLE_DEVICES=2 python ./src/main_lift.py \
    -dataset_test 'h36m' -dataset_train 'h36m' -protocol 1 -noise 0 -canonical \
    -network 'resnet' -mode 4 -num_layers 2 -num_features 4096 \
    -opt_method rmsprop -lr 1e-3 -num_epochs 300 -batch_size 64

# + location & scale
CUDA_VISIBLE_DEVICES=2 python ./src/main_lift.py \
    -dataset_test 'h36m' -dataset_train 'h36m' -protocol 1 -noise 0 -canonical \
    -network 'resnet' -mode 1 -num_layers 2 -num_features 4096 \
    -opt_method rmsprop -lr 1e-3 -num_epochs 300 -batch_size 64

