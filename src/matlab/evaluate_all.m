close all;
clear;
clc;

% Add path for evaluation
addpath('./test_util');

% 3d dataset: H36M
% w/o canonical depth
opt.dataset2d = 'inf_mpii';
opt.dataset3d = 'h36m';
opt.canonical = 0;
evaluate_result(opt);

% 3d dataset: H36M
% w/ canonical depth
opt.dataset2d = 'inf_mpii';
opt.dataset3d = 'h36m';
opt.canonical = 1;
evaluate_result(opt);

% 3d dataset: INF
% w/o canonical depth
opt.dataset2d = 'inf_mpii';
opt.dataset3d = 'inf';
opt.canonical = 0;
evaluate_result(opt);

% 3d dataset: INF
% w/ canonical depth
opt.dataset2d = 'inf_mpii';
opt.dataset3d = 'inf';
opt.canonical = 1;
evaluate_result(opt);

