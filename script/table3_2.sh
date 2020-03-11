#!/bin/bash

# script for table 2

###########################################################################################################
# noise=0: use GT for learning PoseLifter

CUDA_VISIBLE_DEVICES=2 python ./src/main_lift.py \
    -dataset_test 'h36m' -dataset_train 'h36m' -protocol 1 -noise 0 -canonical \
    -network 'resnet' -mode 1 -num_layers 2 -num_features 4096 \
    -opt_method rmsprop -lr 1e-3 -num_epochs 300 -batch_size 64

CUDA_VISIBLE_DEVICES=2 python ./src/main_2d.py \
    -dataset_test 'h36m' -dataset_train 'h36m_mpii' -protocol 1 \
    -network 'resnet152' \
    -model_2d_path 'test_h36m_protocol1/resnet152-int/train_h36m_mpii/rmsprop_lr1.0e-05_batch48_weight1.0e+00/final_model.pth' \
    -lift3d 2 -dataset3d 'h36m' -mode 1 -num_features 4096 -noise 0 \
    -model_lift_path 'test_h36m_protocol1/resnet3dr-canonical/train_h36m_noise0/mode1_nLayer2_nFeat4096_rmsprop_lr1.0e-03_batch64/final_model.pth' \
    -opt_method 'rmsprop' -lr 1e-5 \
    -num_epochs 0 -batch_size 64

###########################################################################################################
# noise=2: use 2D estimate for learning PoseLifter

CUDA_VISIBLE_DEVICES=1,2,3 python ./src/main_2d.py \
    -dataset_test 'h36m' -dataset_train 'h36m_mpii' -protocol 1 -multi_gpu -save_results_train \
    -network 'resnet152' -integral 1 -weight1 1e0 \
    -model_2d_path 'test_h36m_protocol1/resnet152/train_h36m_mpii/rmsprop_lr1.0e-04_batch48/final_model.pth' \
    -opt_method 'rmsprop' -lr 1e-5 \
    -num_epochs 10 -batch_size 48

CUDA_VISIBLE_DEVICES=2 python ./src/main_lift.py \
    -dataset_test 'h36m' -dataset_train 'h36m' -protocol 1 -noise 2 -canonical \
    -noise_path 'test_h36m_protocol1/resnet152-int/train_h36m_mpii/rmsprop_lr1.0e-05_batch48_weight1.0e+00/test_train.pth' \
    -network 'resnet' -mode 1 -num_layers 2 -num_features 4096 \
    -opt_method rmsprop -lr 1e-3 -num_epochs 300 -batch_size 64

CUDA_VISIBLE_DEVICES=2 python ./src/main_2d.py \
    -dataset_test 'h36m' -dataset_train 'h36m_mpii' -protocol 1 \
    -network 'resnet152' \
    -model_2d_path 'test_h36m_protocol1/resnet152-int/train_h36m_mpii/rmsprop_lr1.0e-05_batch48_weight1.0e+00/final_model.pth' \
    -lift3d 2 -dataset3d 'h36m' -mode 1 -num_features 4096 -noise 2 \
    -model_lift_path 'test_h36m_protocol1/resnet3dr-canonical/train_h36m_noise2/mode1_nLayer2_nFeat4096_rmsprop_lr1.0e-03_batch64/final_model.pth' \
    -opt_method 'rmsprop' -lr 1e-5 \
    -num_epochs 0 -batch_size 64

###########################################################################################################
# noise=3: use single Gaussian noise for learning PoseLifter

CUDA_VISIBLE_DEVICES=2 python ./src/main_lift.py \
    -dataset_test 'h36m' -dataset_train 'h36m' -protocol 1 -noise 3 -canonical \
    -noise_path 'test_h36m_protocol0/resnet152-int/train_h36m_mpii/rmsprop_lr1.0e-05_batch48_weight1.0e+00/analysis/stat_simple.pth' \
    -network 'resnet' -mode 1 -num_layers 2 -num_features 4096 \
    -opt_method rmsprop -lr 1e-3 -num_epochs 300 -batch_size 64

CUDA_VISIBLE_DEVICES=2 python ./src/main_2d.py \
    -dataset_test 'h36m' -dataset_train 'h36m_mpii' -protocol 1 \
    -network 'resnet152' \
    -model_2d_path 'test_h36m_protocol1/resnet152-int/train_h36m_mpii/rmsprop_lr1.0e-05_batch48_weight1.0e+00/final_model.pth' \
    -lift3d 2 -dataset3d 'h36m' -mode 1 -num_features 4096 -noise 3 \
    -model_lift_path 'test_h36m_protocol1/resnet3dr-canonical/train_h36m_noise3/mode1_nLayer2_nFeat4096_rmsprop_lr1.0e-03_batch64/final_model.pth' \
    -opt_method 'rmsprop' -lr 1e-5 \
    -num_epochs 0 -batch_size 64

###########################################################################################################
# noise=4: use proposed mixture noise for learning PoseLifter

CUDA_VISIBLE_DEVICES=2 python ./src/main_lift.py \
    -dataset_test 'h36m' -dataset_train 'h36m' -protocol 1 -noise 4 -canonical \
    -noise_path 'test_h36m_protocol0/resnet152-int/train_h36m_mpii/rmsprop_lr1.0e-05_batch48_weight1.0e+00/analysis/stat_2d_em1.pth' \
    -network 'resnet' -mode 1 -num_layers 2 -num_features 4096 \
    -opt_method rmsprop -lr 1e-3 -num_epochs 300 -batch_size 64

CUDA_VISIBLE_DEVICES=2 python ./src/main_2d.py \
    -dataset_test 'h36m' -dataset_train 'h36m_mpii' -protocol 1 \
    -network 'resnet152' \
    -model_2d_path 'test_h36m_protocol1/resnet152-int/train_h36m_mpii/rmsprop_lr1.0e-05_batch48_weight1.0e+00/final_model.pth' \
    -lift3d 2 -dataset3d 'h36m' -mode 1 -num_features 4096 -noise 4 \
    -model_lift_path 'test_h36m_protocol1/resnet3dr-canonical/train_h36m_noise4/mode1_nLayer2_nFeat4096_rmsprop_lr1.0e-03_batch64/final_model.pth' \
    -opt_method 'rmsprop' -lr 1e-5 \
    -num_epochs 0 -batch_size 64

