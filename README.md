# PoseLifter: Absolute 3D human pose lifting network from a single noisy 2D human pose

## Introduction
This repo provides the source code to reproduce all the results of our [arXiv paper](https://arxiv.org/abs/1910.12029):

> Ju Yong Chang, Gyeongsik Moon, Kyoung Mu Lee. PoseLifter: Absolute 3D human pose lifting network from a single noisy 2D human pose. arXiv preprint arXiv:1910.12029, 2019.

## Dataset setup
You can download the [annotation data](https://drive.google.com/open?id=1ldmoS7h49Ww-4pdAMTtdFmpoyz9nwSKV) and the images for [Human3.6M](https://drive.google.com/open?id=1A_cRrGaUjFRywECQR0Sn7AMWU6LSeLkc), [MPI-INF-3DHP](https://drive.google.com/open?id=1IxxGvFVgJ52JF1emEfjxOQoIQvBIhlxU), and [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/).

## Reproducing our results
You can reproduce the results of our paper by running the script files in the script directory. For example, to generate the numbers in Table 2, run:
```bash
./script/table2.sh
```

