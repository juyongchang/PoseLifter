import argparse
import os
import conf
import pdb

class Opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
    
    def init(self):
        # miscellaneous
        self.parser.add_argument('-dataset_test', default='h36m', help='Dataset to test: h36m | inf')
        self.parser.add_argument('-dataset_train', default='h36m', help='Dataset to train: h36m | mpii | h36m_mpii')
        self.parser.add_argument('-protocol', type=int, default=1, help='Experiment protocol: 0| 1 | 2')
        self.parser.add_argument('-multi_gpu', default=False, action='store_true', help='Use multiple gpus?')
        self.parser.add_argument('-fliptest', default=False, action='store_true', help='Do flip test?')
        self.parser.add_argument('-save_results', default=False, action='store_true', help='Save prediction results?')
        self.parser.add_argument('-save_results_train', default=False, action='store_true', help='Save prediction results?')

        # network structure (2d)
        self.parser.add_argument('-network', default='resnet50 | resnet101 | resnet152', help='Network to use')
        self.parser.add_argument('-integral', type=int, default=0, help='Use the integral module?')
        self.parser.add_argument('-weight1', type=float, default=1.0, help='Weight for coordinate loss')
        self.parser.add_argument('-model_2d_path', default=None, help='Path to pretrained model')

        # network structure (3d)
        self.parser.add_argument('-lift3d', type=int, default=0, help='Lift to 3D?')
        self.parser.add_argument('-dataset3d', default='h36m', help='Dataset for 3D lifting')
        self.parser.add_argument('-weight2', type=float, default=1.0, help='Weight for 3d loss')
        self.parser.add_argument('-mode', type=int, default=1, help='Option for lifting')
        self.parser.add_argument('-num_layers', type=int, default=2, help='Number of hidden layers')
        self.parser.add_argument('-num_features', type=int, default=3000, help='Number of features')
        self.parser.add_argument('-noise', type=int, default=1, help='Noise mode')
        self.parser.add_argument('-std_train', type=float, default=0.005, help='Std of Gaussian noise for robust training')
        self.parser.add_argument('-model_lift_path', default=None, help='Path to pretrained model')

        # optimization hyperparameters
        self.parser.add_argument('-opt_method', default='rmsprop', help='Optimization method: rmsprop | adam | sgd')
        self.parser.add_argument('-lr', type=float, default=1e-4, help='Learning rate')
        self.parser.add_argument('-alpha', type=float, default=0.99, help='Smoothing constant')
        self.parser.add_argument('-epsilon', type=float, default=1e-8, help='For numerical stability')
        self.parser.add_argument('-weight_decay', type=float, default=0, help='Weight decay')
        self.parser.add_argument('-lr_decay', type=float, default=0, help='Learning rate decay')
        self.parser.add_argument('-beta1', type=float, default=0.9, help='First mement coefficient')
        self.parser.add_argument('-beta2', type=float, default=0.99, help='Second moment coefficient')
        self.parser.add_argument('-momentum', type=float, default=0, help='Momentum')

        # training options
        self.parser.add_argument('-num_epochs', type=int, default=30, help='Number of training epochs')
        self.parser.add_argument('-batch_size', type=int, default=8, help='Mini-batch size')
        self.parser.add_argument('-save_intervals', type=int, default=10, help='Number of iterations for saving model')
    
    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

        # set directories for experiments
        if self.opt.dataset_test == 'h36m':
            self.opt.save_dir = '%s/test_%s_protocol%d' % (conf.exp_dir, self.opt.dataset_test, self.opt.protocol)
        elif self.opt.dataset_test == 'inf':
            self.opt.save_dir = '%s/test_%s' % (conf.exp_dir, self.opt.dataset_test)
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)

        if self.opt.lift3d == 0: # no 3d lifting
            if self.opt.integral == 0:
                self.opt.save_dir = os.path.join(self.opt.save_dir, '%s' % self.opt.network)
                if not os.path.exists(self.opt.save_dir):
                    os.makedirs(self.opt.save_dir)
                self.opt.save_dir = os.path.join(self.opt.save_dir, 'train_%s' % self.opt.dataset_train)
                if not os.path.exists(self.opt.save_dir):
                    os.makedirs(self.opt.save_dir)
                self.opt.save_dir = os.path.join(self.opt.save_dir, '%s_lr%1.1e_batch%d' % \
                    (self.opt.opt_method, self.opt.lr, self.opt.batch_size))
            else:
                self.opt.save_dir = os.path.join(self.opt.save_dir, '%s-int' % (self.opt.network))
                if not os.path.exists(self.opt.save_dir):
                    os.makedirs(self.opt.save_dir)
                self.opt.save_dir = os.path.join(self.opt.save_dir, 'train_%s' % self.opt.dataset_train)
                if not os.path.exists(self.opt.save_dir):
                    os.makedirs(self.opt.save_dir)
                self.opt.save_dir = os.path.join(self.opt.save_dir, '%s_lr%1.1e_batch%d_weight%1.1e' % \
                    (self.opt.opt_method, self.opt.lr, self.opt.batch_size, self.opt.weight1))

        else: # piecewise lifting (integrated 2d pose -> 3d pose + root depth)
            self.opt.save_dir = os.path.join(self.opt.save_dir, '%s-lift' % (self.opt.network))
            if not os.path.exists(self.opt.save_dir):
                os.makedirs(self.opt.save_dir)
            self.opt.save_dir = os.path.join(self.opt.save_dir, 'train2d_%s_train3d_%s' % (self.opt.dataset_train, self.opt.dataset3d))
            if not os.path.exists(self.opt.save_dir):
                os.makedirs(self.opt.save_dir)
            if self.opt.noise != 1:
                self.opt.save_dir = os.path.join(self.opt.save_dir, 'canonical%d_mode%d_noise%d' % (int(self.opt.lift3d==2), self.opt.mode, self.opt.noise))
            else:
                self.opt.save_dir = os.path.join(self.opt.save_dir, 'canonical%d_mode%d_noise%d_std%.3f' % (int(self.opt.lift3d==2), self.opt.mode, self.opt.noise, self.opt.std_train))

        # save options
        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))
        confs = dict((name, getattr(conf, name)) for name in dir(conf)
                    if not name.startswith('_'))
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)
        file_name = os.path.join(self.opt.save_dir, 'opt.txt')
        if not os.path.exists(file_name):
            with open(file_name, 'wt') as opt_file:
                opt_file.write('==> Args:\n')
                for k, v in sorted(args.items()):
                    opt_file.write('  %s: %s\n' % (str(k), str(v)))
                opt_file.write('==> Args:\n')
                for k, v in sorted(confs.items()):
                    opt_file.write('  %s: %s\n' % (str(k), str(v)))
        
        return self.opt

