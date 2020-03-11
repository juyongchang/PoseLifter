#!/bin/bash

# script for table 5

###########################################################################################################
# 2D pose estimation

CUDA_VISIBLE_DEVICES=1,2,3 python ./src/main_2d.py \
    -dataset_test 'inf' -dataset_train 'inf_mpii' -multi_gpu \
    -network 'resnet152' \
    -opt_method 'rmsprop' -lr 1e-4 \
    -num_epochs 50 -batch_size 48

CUDA_VISIBLE_DEVICES=1,2,3 python ./src/main_2d.py \
    -dataset_test 'inf' -dataset_train 'inf_mpii' -multi_gpu \
    -network 'resnet152' -integral 1 -weight1 1e0 \
    -model_2d_path 'test_inf/resnet152/train_inf_mpii/rmsprop_lr1.0e-04_batch48/final_model.pth' \
    -opt_method 'rmsprop' -lr 1e-5 \
    -num_epochs 10 -batch_size 48

###########################################################################################################
# 3d dataset: H36M
# w/o canonical depth

CUDA_VISIBLE_DEVICES=2 python ./src/main_lift.py \
    -dataset_test 'h36m' -dataset_train 'h36m' -protocol 1 -noise 4 -scale \
    -noise_path 'test_h36m_protocol0/resnet152-int/train_h36m_mpii/rmsprop_lr1.0e-05_batch48_weight1.0e+00/analysis/stat_2d_em1.pth' \
    -network 'resnet' -mode 1 -num_layers 2 -num_features 4096 \
    -opt_method 'rmsprop' -lr 1e-3 -num_epochs 300 -batch_size 64

CUDA_VISIBLE_DEVICES=2 python ./src/main_2d.py \
    -dataset_test 'inf' -dataset_train 'inf_mpii' -fliptest -save_results \
    -network 'resnet152' \
    -model_2d_path 'test_inf/resnet152-int/train_inf_mpii/rmsprop_lr1.0e-05_batch48_weight1.0e+00/final_model.pth' \
    -lift3d 1 -dataset3d 'h36m' -mode 1 -num_features 4096 -noise 4 \
    -model_lift_path 'test_h36m_protocol1/resnet3dr/train_h36m_scale_noise4/mode1_nLayer2_nFeat4096_rmsprop_lr1.0e-03_batch64/final_model.pth' \
    -opt_method 'rmsprop' -lr 1e-5 \
    -num_epochs 0 -batch_size 64

###########################################################################################################
# 3d dataset: H36M
# w/ canonical depth

CUDA_VISIBLE_DEVICES=2 python ./src/main_lift.py \
    -dataset_test 'h36m' -dataset_train 'h36m' -protocol 1 -noise 4 -canonical -scale \
    -noise_path 'test_h36m_protocol0/resnet152-int/train_h36m_mpii/rmsprop_lr1.0e-05_batch48_weight1.0e+00/analysis/stat_2d_em1.pth' \
    -network 'resnet' -mode 1 -num_layers 2 -num_features 4096 \
    -opt_method 'rmsprop' -lr 1e-3 -num_epochs 300 -batch_size 64

CUDA_VISIBLE_DEVICES=2 python ./src/main_2d.py \
    -dataset_test 'inf' -dataset_train 'inf_mpii' -fliptest -save_results \
    -network 'resnet152' \
    -model_2d_path 'test_inf/resnet152-int/train_inf_mpii/rmsprop_lr1.0e-05_batch48_weight1.0e+00/final_model.pth' \
    -lift3d 2 -dataset3d 'h36m' -mode 1 -num_features 4096 -noise 4 \
    -model_lift_path 'test_h36m_protocol1/resnet3dr-canonical/train_h36m_scale_noise4/mode1_nLayer2_nFeat4096_rmsprop_lr1.0e-03_batch64/final_model.pth' \
    -opt_method 'rmsprop' -lr 1e-5 \
    -num_epochs 0 -batch_size 64

###########################################################################################################
# 3d dataset: INF
# w/o canonical depth

CUDA_VISIBLE_DEVICES=2 python ./src/main_lift.py \
    -dataset_test 'inf' -dataset_train 'inf' -noise 4 \
    -noise_path 'test_h36m_protocol0/resnet152-int/train_h36m_mpii/rmsprop_lr1.0e-05_batch48_weight1.0e+00/analysis/stat_2d_em1.pth' \
    -network 'resnet' -mode 1 -num_layers 2 -num_features 4096 \
    -opt_method 'rmsprop' -lr 1e-3 -num_epochs 300 -batch_size 64

CUDA_VISIBLE_DEVICES=2 python ./src/main_2d.py \
    -dataset_test 'inf' -dataset_train 'inf_mpii' -fliptest -save_results \
    -network 'resnet152' \
    -model_2d_path 'test_inf/resnet152-int/train_inf_mpii/rmsprop_lr1.0e-05_batch48_weight1.0e+00/final_model.pth' \
    -lift3d 1 -dataset3d 'inf' -mode 1 -num_features 4096 -noise 4 \
    -model_lift_path 'test_inf/resnet3dr/train_inf_noise4/mode1_nLayer2_nFeat4096_rmsprop_lr1.0e-03_batch64/final_model.pth' \
    -opt_method 'rmsprop' -lr 1e-5 \
    -num_epochs 0 -batch_size 64

###########################################################################################################
# 3d dataset: INF
# w/ canonical depth

CUDA_VISIBLE_DEVICES=2 python ./src/main_lift.py \
    -dataset_test 'inf' -dataset_train 'inf' -noise 4 -canonical \
    -noise_path 'test_h36m_protocol0/resnet152-int/train_h36m_mpii/rmsprop_lr1.0e-05_batch48_weight1.0e+00/analysis/stat_2d_em1.pth' \
    -network 'resnet' -mode 1 -num_layers 2 -num_features 4096 \
    -opt_method 'rmsprop' -lr 1e-3 -num_epochs 300 -batch_size 64

CUDA_VISIBLE_DEVICES=2 python ./src/main_2d.py \
    -dataset_test 'inf' -dataset_train 'inf_mpii' -fliptest -save_results \
    -network 'resnet152' \
    -model_2d_path 'test_inf/resnet152-int/train_inf_mpii/rmsprop_lr1.0e-05_batch48_weight1.0e+00/final_model.pth' \
    -lift3d 2 -dataset3d 'inf' -mode 1 -num_features 4096 -noise 4 \
    -model_lift_path 'test_inf/resnet3dr-canonical/train_inf_noise4/mode1_nLayer2_nFeat4096_rmsprop_lr1.0e-03_batch64/final_model.pth' \
    -opt_method 'rmsprop' -lr 1e-5 \
    -num_epochs 0 -batch_size 64

