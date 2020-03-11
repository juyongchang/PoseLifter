#!/bin/bash

# script for table 2

###########################################################################################################
# 2D pose estimation for protocol 0

CUDA_VISIBLE_DEVICES=1,2,3 python ./src/main_2d.py \
    -dataset_test 'h36m' -dataset_train 'h36m_mpii' -protocol 0 -multi_gpu \
    -network 'resnet152' \
    -opt_method 'rmsprop' -lr 1e-4 \
    -num_epochs 50 -batch_size 48

CUDA_VISIBLE_DEVICES=1,2,3 python ./src/main_2d.py \
    -dataset_test 'h36m' -dataset_train 'h36m_mpii' -protocol 0 -multi_gpu -save_results \
    -network 'resnet152' -integral 1 -weight1 1e0 \
    -model_2d_path 'test_h36m_protocol0/resnet152/train_h36m_mpii/rmsprop_lr1.0e-04_batch48/final_model.pth' \
    -opt_method 'rmsprop' -lr 1e-5 \
    -num_epochs 10 -batch_size 48

###########################################################################################################
# Error analysis for protocol 0

python ./src/analysis_error_mixture.py

###########################################################################################################
# 2D pose estimation for protocol 1

CUDA_VISIBLE_DEVICES=1,2,3 python ./src/main_2d.py \
    -dataset_test 'h36m' -dataset_train 'h36m_mpii' -protocol 1 -multi_gpu \
    -network 'resnet152' \
    -opt_method 'rmsprop' -lr 1e-4 \
    -num_epochs 50 -batch_size 48

CUDA_VISIBLE_DEVICES=1,2,3 python ./src/main_2d.py \
    -dataset_test 'h36m' -dataset_train 'h36m_mpii' -protocol 1 -multi_gpu \
    -network 'resnet152' -integral 1 -weight1 1e0 \
    -model_2d_path 'test_h36m_protocol1/resnet152/train_h36m_mpii/rmsprop_lr1.0e-04_batch48/final_model.pth' \
    -opt_method 'rmsprop' -lr 1e-5 \
    -num_epochs 10 -batch_size 48

