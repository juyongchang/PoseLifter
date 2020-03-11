import os
import torch
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib
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

# set font
font = {'family' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

#----------------------------------------------------------------------
# compute error
error = pred - gt

# create save directory
save_dir = '%s/analysis/stat_2d_em1' % (exp_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#----------------------------------------------------------------------
# parameter estimation using EM
mean = np.zeros((num_joint, 2))
std = np.zeros((num_joint, 2))
weight = np.zeros((num_joint))
for j in range(num_joint):
    e = error[:, j, :]

    # non-robust estimation of std (we assume zero mean)
    m1 = np.mean(e, axis=0)
    s1 = np.sqrt(np.mean((e-m1)**2.0, axis=0))

    # robust estimation of std (we assume zero mean)
    q75 = np.percentile(e, 75, axis=0)
    q25 = np.percentile(e, 25, axis=0)
    m2 = np.percentile(e, 50, axis=0)
    s2 = 0.7413 * (q75 - q25)

    # initial estimate
    m = m2
    s = s2
    w = 0.5
    print(m, s, w)

    # NLL log
    NLL = np.zeros((10))

    for k in range(10):
        # E-step: compute responsibility
        p1 = w * np.exp(-np.sum(((e-m)**2.0)/(2*s**2.0), axis=1, keepdims=True))/(2*np.pi*s[0]*s[1])
        p2 = (1.0-w) * 1/10000.0
        r1 = p1 / (p1 + p2)
        r2 = p2 / (p1 + p2)

        # M-step: re-estimate the parameters
        m = np.sum(r1 * e, axis=0) / np.sum(r1)
        s = np.sqrt(np.sum(r1*(e-m)**2.0, axis=0)/np.sum(r1))
        w = np.mean(r1)

        # compute NLL
        nll = -np.mean(np.log(p1+p2))
        NLL[k] = nll

        #
        print(m, s, w, nll)

    #
    mean[j] = m
    std[j] = s
    weight[j] = w

    #
    tick_x_val = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tick_x_lab = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    plt.clf()
    plt.plot(np.linspace(1,10,10), NLL, '-s', lw=1.5, markeredgecolor='k', markerfacecolor='r')
    plt.xlabel('Iteration')
    plt.ylabel('Negative Log Likelihood')
    plt.title('%s' % (name_joint[j]))
    plt.xticks(tick_x_val, tick_x_lab)
    plt.grid(True)
    plt.savefig('%s/em_%s.pdf' % (save_dir, name_joint[j]))

    #
    for i in range(2):
        pdf = lambda x: np.exp(-((x-m[i])**2.0)/(2*s[i]**2.0))*w/np.sqrt(2*np.pi*s[i]**2.0)+(1-w)/100.0

        tick_x_val = [-20, -10, 0, 10, 20]
        tick_x_lab = ['-20', '-10', '0', '10', '20']

        plt.clf()
        plt.hist(e[:, i], bins=100, histtype='stepfilled', range=[-20.0, 20.0], normed=True)

        x_sample = np.linspace(-20, 20, 1000)
        plt.plot(x_sample, norm(m1[i], s1[i]).pdf(x_sample), '--g', lw=1.5,
                 label='Gaussian Fit')
        #plt.plot(x_sample, norm(m2[i], s2[i]).pdf(x_sample), ':k', lw=1.5,
        #         label='robust gaussian fit')
        plt.plot(x_sample, pdf(x_sample), '-r', lw=1.5,
                 label='Mixture Fit')
        plt.legend()

        plt.xlabel('Error (in pixels)')
        plt.ylabel('Probability')
        if i == 0:
            plt.title('%s (x)' % (name_joint[j]))
        elif i == 1:
            plt.title('%s (y)' % (name_joint[j]))
        plt.xticks(tick_x_val, tick_x_lab)
        plt.grid(True)
        if i == 0:
            plt.savefig('%s/error_%s_x.pdf' % (save_dir, name_joint[j]))
        elif i == 1:
            plt.savefig('%s/error_%s_y.pdf' % (save_dir, name_joint[j]))

# save results
print(mean)
print(std)
print(weight)
mean = torch.from_numpy(mean)
std = torch.from_numpy(std)
weight = torch.from_numpy(weight)
stat = {'mean': mean, 'std': std, 'weight': weight}
filename = '%s/analysis/stat_2d_em1.pth' % (exp_dir)
torch.save(stat, filename)

# save txt results
file = open('%s/analysis/stat_2d_em1.txt' % (exp_dir), 'w')
for i in range(num_joint):
    file.write('============================\n')
    file.write('Joint: %s\n' % (name_joint[i]))
    file.write('mean = (%.02f, %.02f)\n' % (mean[i,0], mean[i,1]))
    file.write('std = (%.02f, %.02f)\n' % (std[i,0], std[i,1]))
    file.write('weight = (%.02f)\n' % (weight[i]))
file.close()

