import os
import torch
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pdb

import conf

#----------------------------------------------------------------------
# options
dataset_test = 'h36m'
dataset_train = 'h36m_mpii'
protocol = 0

# joint names
name_joint = ['Hip', \
    'RightHip', 'RightKnee', 'RightFoot', \
    'LeftHip', 'LeftKnee', 'LeftFoot', \
    'Spine', 'Neck', 'Head', 'Site', \
    'LeftShoulder', 'LeftElbow', 'LeftHand', \
    'RightShoulder', 'RightElbow', 'RightHand']

# experiment directory
exp_dir = '%s/test_%s_protocol%d/resnet152-int/train_%s/rmsprop_lr1.0e-05_batch48_weight1.0e+00' \
    % (conf.exp_dir, dataset_test, protocol, dataset_train)

# load result file
filename = '%s/result.pth' % (exp_dir)
result = torch.load(filename)
pred = result['pred2d'].cpu().numpy()
gt = result['gt2d'].cpu().numpy()

# number of samples
num_sample = pred.shape[0]

# number of joints
num_joint = pred.shape[1]

#----------------------------------------------------------------------
# compute error
error = pred - gt

# stats
mean = np.zeros((3, num_joint, 2))
std = np.zeros((3, num_joint, 2))

# non-robust estimation (we assume zero mean)
for j in range(num_joint):
    for i in range(2):
        mean[0, j, i] = np.mean(error[:, j, i])
        std[0, j, i] = np.sqrt(np.mean(error[:, j, i] ** 2.0))

# robust estimation of std (we assume zero mean)
std1 = np.zeros((num_joint, 2))
for j in range(num_joint):
    for i in range(2):
        mean[1, j, i] = np.median(error[:, j, i])
        e = np.absolute(error[:, j, i] - mean[1, j, i])
        m = np.median(e)
        std[1, j, i] = m * 1.4826

# robust estimation of std (we assume zero mean)
std2 = np.zeros((num_joint, 2))
for j in range(num_joint):
    for i in range(2):
        e = error[:, j, i]
        q75 = np.percentile(e, 75)
        q25 = np.percentile(e, 25)
        mean[2, j, i] = np.percentile(e, 50)
        std[2, j, i] = 0.7413 * (q75 - q25)

# save std
print(mean[0])
print(std[0])
print(mean[1])
print(std[1])
print(mean[2])
print(std[2])
mean = torch.from_numpy(mean)
std = torch.from_numpy(std)
stat = {'mean': mean, 'std': std}
filename = '%s/analysis/stat_simple.pth' % (exp_dir)
torch.save(stat, filename)

#----------------------------------------------------------------------
# plot error histogram
save_dir = '%s/analysis/stat_simple' % (exp_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for j in range(num_joint):
    for i in range(2):
        e = error[:, j, i]

        plt.clf()
        plt.hist(e, bins=100, histtype='stepfilled', range=[-20.0, 20.0], normed=True)

        x_sample = np.linspace(-20, 20, 1000)
        plt.plot(x_sample, norm(mean[0, j, i], std[0, j, i]).pdf(x_sample), '-k', lw=1.5,
                 label='non-robust fit')
        plt.plot(x_sample, norm(mean[1, j, i], std[1, j, i]).pdf(x_sample), '--k', lw=1.5,
                 label='robust fit - mad')
        plt.plot(x_sample, norm(mean[2, j, i], std[2, j, i]).pdf(x_sample), ':k', lw=1.5,
                 label='robust fit - iqr')
        plt.legend()

        plt.xlabel('Error (in pixels)')
        plt.ylabel('Numbers')
        plt.title('Mean(%.2f), Std0(%.2f), Std1(%.2f), Std2(%.2f)' % (e.mean(), std[0, j, i], std[1, j, i], std[2, j, i]))
        plt.grid(True)
        if i == 0:
            plt.savefig('%s/error_%s_x.png' % (save_dir, name_joint[j]))
        elif i == 1:
            plt.savefig('%s/error_%s_y.png' % (save_dir, name_joint[j]))

