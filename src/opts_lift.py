import argparse
import os
import conf
import pdb

class Opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
    
    def init(self):
        # miscellaneous
        self.parser.add_argument('-dataset_test', default='h36m', help='Test dataset')
        self.parser.add_argument('-dataset_train', default='h36m', help='Training dataset')
        self.parser.add_argument('-protocol', type=int, default=1, help='Experiment protocol for H36M: 0 | 1 | 2')
        self.parser.add_argument('-multi_gpu', default=False, action='store_true', help='Use multiple gpus?')
        self.parser.add_argument('-noise', type=int, default=0, help='Noise mode')
        self.parser.add_argument('-noise_path', default=None, help='Path to noise info')
        self.parser.add_argument('-std_train', type=float, default=0.0, help='Std of Gaussian noise for robust training')
        self.parser.add_argument('-std_test', type=float, default=0.0, help='Std of Gaussian noise for testing')
        self.parser.add_argument('-canonical', default=False, action='store_true', help='Use canonical coordinate for root?')
        self.parser.add_argument('-scale', default=False, action='store_true', help='Induce random scaling for data augmentation?')
        self.parser.add_argument('-fliptest', default=False, action='store_true', help='Do flip test?')
        self.parser.add_argument('-analysis', default=False, action='store_true', help='Analyze results?')

        # network structure
        self.parser.add_argument('-network', default='resnet', help='Network to use: resnet')
        self.parser.add_argument('-mode', type=int, default='1', help='Use location and scale info?')
        self.parser.add_argument('-num_layers', type=int, default=2, help='Number of hidden layers')
        self.parser.add_argument('-num_features', type=int, default=4096, help='Number of features')
        self.parser.add_argument('-weight_root', type=float, default=1.0, help='Weight for root loss')

        # optimization
        self.parser.add_argument('-opt_method', default='rmsprop', help='Optimization method: rmsprop | adam')
        self.parser.add_argument('-lr', type=float, default=1e-3, help='Learning rate')
        self.parser.add_argument('-alpha', type=float, default=0.99, help='Smoothing constant')
        self.parser.add_argument('-epsilon', type=float, default=1e-8, help='For numerical stability')
        self.parser.add_argument('-weight_decay', type=float, default=0, help='Weight decay')
        self.parser.add_argument('-lr_decay', type=float, default=0, help='Learning rate decay')
        self.parser.add_argument('-beta1', type=float, default=0.9, help='First mement coefficient')
        self.parser.add_argument('-beta2', type=float, default=0.99, help='Second moment coefficient')
        self.parser.add_argument('-momentum', type=float, default=0, help='Momentum')

        # training options
        self.parser.add_argument('-num_epochs', type=int, default=200, help='Number of training epochs')
        self.parser.add_argument('-batch_size', type=int, default=512, help='Mini-batch size')
        self.parser.add_argument('-save_intervals', type=int, default=50, help='Number of iterations for saving model')
    
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

        if self.opt.canonical == False:
            self.opt.save_dir = os.path.join(self.opt.save_dir, '%s3dr' % self.opt.network)
        else:
            self.opt.save_dir = os.path.join(self.opt.save_dir, '%s3dr-canonical' % self.opt.network)
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)

        if self.opt.noise != 1:
            if self.opt.scale == False:
                self.opt.save_dir = os.path.join(self.opt.save_dir, 'train_%s_noise%d' % (self.opt.dataset_train, self.opt.noise))
            else:
                self.opt.save_dir = os.path.join(self.opt.save_dir, 'train_%s_scale_noise%d' % (self.opt.dataset_train, self.opt.noise))
        else:
            if self.opt.scale == False:
                self.opt.save_dir = os.path.join(self.opt.save_dir, 'train_%s_noise%d_std%.3f' % (self.opt.dataset_train, self.opt.noise, self.opt.std_train))
            else:
                self.opt.save_dir = os.path.join(self.opt.save_dir, 'train_%s_scale_noise%d_std%.3f' % (self.opt.dataset_train, self.opt.noise, self.opt.std_train))
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)

        self.opt.save_dir = '%s/mode%d_nLayer%d_nFeat%d_%s_lr%1.1e_batch%d' % (self.opt.save_dir, self.opt.mode, self.opt.num_layers, self.opt.num_features, self.opt.opt_method, self.opt.lr, self.opt.batch_size)
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)

        # save options
        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))
        refs = dict((name, getattr(conf, name)) for name in dir(conf)
                    if not name.startswith('_'))
        file_name = os.path.join(self.opt.save_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')
            for k, v in sorted(refs.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
        
        return self.opt

