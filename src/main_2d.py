import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import pdb

import conf
from opts_2d import Opts
from models.resnet2d import ResNet
from models.resnet2d_int import ResNetInt
from models.resnet2d_lift import ResNetLift
from datasets_2d.h36m17 import H36M17
from datasets_2d.mpii import MPII
from datasets_2d.h36m17_mpii import H36M17_MPII
from datasets_2d.mpiinf import MPIINF
from datasets_2d.mpiinf_mpii import MPIINF_MPII
from train_2d import train, val, test, test_train
from utils.logger import Logger

def main():
    # for repeatable experiments
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # options
    opt = Opts().parse()

    # model
    if opt.lift3d == 0:
        if opt.integral == 0:
            model = ResNet(opt.network, conf.num_joints).cuda()
        else:
            if opt.model_2d_path is None:
                model = ResNetInt(opt.network, conf.num_joints).cuda()
            else:
                filename = '%s/%s' % (conf.exp_dir, opt.model_2d_path)
                model = ResNetInt(opt.network, conf.num_joints, filename).cuda()
    else:
        filename_2d = '%s/%s' % (conf.exp_dir, opt.model_2d_path)
        filename_lift = '%s/%s' % (conf.exp_dir, opt.model_lift_path)
        model = ResNetLift(opt.network, conf.num_joints, opt.num_layers, opt.num_features, opt.mode, \
            filename_2d, filename_lift).cuda()

    # multi-gpu
    if opt.multi_gpu == True:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
    else:
        model = torch.nn.DataParallel(model, device_ids=[0])

    # dataset loader (train)
    if opt.dataset_train == 'h36m':
        train_loader = torch.utils.data.DataLoader(
            H36M17(opt.protocol, 'train'),
            batch_size = opt.batch_size,
            shuffle = True,
            num_workers = int(conf.num_threads))
    elif opt.dataset_train == 'mpii':
        train_loader = torch.utils.data.DataLoader(
            MPII(),
            batch_size = opt.batch_size,
            shuffle = True,
            num_workers = int(conf.num_threads))
    elif opt.dataset_train == 'h36m_mpii':
        train_loader = torch.utils.data.DataLoader(
            H36M17_MPII(opt.protocol, 'train'),
            batch_size = opt.batch_size,
            shuffle = True,
            num_workers = int(conf.num_threads))
    elif opt.dataset_train == 'inf':
        train_loader = torch.utils.data.DataLoader(
            MPIINF('train'),
            batch_size = opt.batch_size,
            shuffle = True,
            num_workers = int(conf.num_threads))
    elif opt.dataset_train == 'inf_mpii':
        train_loader = torch.utils.data.DataLoader(
            MPIINF_MPII('train'),
            batch_size = opt.batch_size,
            shuffle = True,
            num_workers = int(conf.num_threads))
    else:
        raise ValueError('unsupported dataset %s' % opt.dataset_train)

    # dataset loader (val)
    if opt.dataset_test == 'h36m':
        val_loader = torch.utils.data.DataLoader(
            H36M17(opt.protocol, 'val'),
            batch_size = opt.batch_size,
            shuffle = False,
            num_workers = int(conf.num_threads))
    elif opt.dataset_test == 'inf':
        val_loader = torch.utils.data.DataLoader(
            MPIINF('val'),
            batch_size = opt.batch_size,
            shuffle = False,
            num_workers = int(conf.num_threads))
    else:
        raise ValueError('unsupported dataset %s' % opt.dataset_test)

    # optimizer
    if opt.opt_method == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr = opt.lr)
    elif opt.opt_method == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr = opt.lr)
    else:
        raise ValueError('unsupported optimizer %s' % opt.opt_method)

    # scheduler
    if opt.lift3d == 0:
        if opt.integral == 0:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)

    # log
    log = []
    log.append([]) # epoch
    log.append([]) # cost (train)
    log.append([]) # error2d1 (train)
    log.append([]) # error2d2 (train)
    log.append([]) # cost (val)
    log.append([]) # error2d1 (val)
    log.append([]) # error2d2 (val)

    # train
    if opt.lift3d == 0:
        idx_start = opt.num_epochs
        while idx_start > 0:
            file_name = os.path.join(opt.save_dir, 'model_{}.pth'.format(idx_start))
            if os.path.exists(file_name):
                state = torch.load(file_name)
                model.load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
                scheduler.load_state_dict(state['scheduler'])
                log_name = os.path.join(opt.save_dir, 'log_{}.pkl'.format(idx_start))
                if os.path.exists(log_name):
                    with open(log_name, 'rb') as fin:
                        log = pickle.load(fin)
                break
            idx_start -= 1

        # logger
        if idx_start == 0:
            logger_error = Logger(opt.save_dir + '/logs_error')
        else:
            logger_error = Logger(opt.save_dir + '/logs_error', reset=False)

        for epoch in range(idx_start+1, opt.num_epochs+1):
            # for repeatable experiments
            np.random.seed(epoch)
            torch.manual_seed(epoch)
            torch.cuda.manual_seed(epoch)

            # do scheduler
            scheduler.step()

            # perform one epoch of training
            cost_train, error2d_train, error3d_train, error3d_pa_train = train(epoch, opt, train_loader, model, optimizer)
            logger_error.scalar_summary('cost_train', cost_train, epoch)
            for i in range(2):
                logger_error.scalar_summary('error2d%d_train' % i, error2d_train[i], epoch)
            logger_error.scalar_summary('error3d_train', error3d_train, epoch)
            logger_error.scalar_summary('error3d_pa_train', error3d_pa_train, epoch)
            
            # perform one epoch of validation
            with torch.no_grad():
                cost_val, error2d_val, error3d_val, error3d_pa_val = val(epoch, opt, val_loader, model)
            logger_error.scalar_summary('cost_val', cost_val, epoch)
            for i in range(2):
                logger_error.scalar_summary('error2d%d_val' % i, error2d_val[i], epoch)
            logger_error.scalar_summary('error3d_val', error3d_val, epoch)
            logger_error.scalar_summary('error3d_pa_val', error3d_pa_val, epoch)

            # print message to log file
            msg = '%d %1.1e' % (epoch, optimizer.param_groups[0]['lr'])
            msg = '%s | %.4f' % (msg, cost_train)
            for i in range(2): msg = '%s %.4f' % (msg, error2d_train[i])
            msg = '%s %.4f %.4f' % (msg, error3d_train, error3d_pa_train)
            msg = '%s | %.4f' % (msg, cost_val)
            for i in range(2): msg = '%s %.4f' % (msg, error2d_val[i])
            msg = '%s %.4f %.4f' % (msg, error3d_val, error3d_pa_val)
            logger_error.write('%s\n' % msg)
            
            #
            log[0].append(epoch)
            log[1].append(cost_train)
            log[2].append(error2d_train[0])
            log[3].append(error2d_train[1])
            log[4].append(cost_val)
            log[5].append(error2d_val[0])
            log[6].append(error2d_val[1])

            # save model
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            if epoch % opt.save_intervals == 0:
                torch.save(state, os.path.join(opt.save_dir, 'model_%d.pth' % (epoch)))
                log_name = os.path.join(opt.save_dir, 'log_%d.pkl' % (epoch))
                with open(log_name, 'wb') as fout:
                    pickle.dump(log, fout)
        
        logger_error.close()

        # save final model
        file_name = os.path.join(opt.save_dir, 'final_model.pth')
        torch.save(state, file_name)

        # save final log
        log_name = os.path.join(opt.save_dir, 'final_log.pkl')
        with open(log_name, 'wb') as fout:
            pickle.dump(log, fout)

        # plotting
        x = range(1, opt.num_epochs+1)
        cost_train = np.array(log[1])
        error2d1_train = np.array(log[2])
        error2d2_train = np.array(log[3])
        cost_val = np.array(log[4])
        error2d1_val = np.array(log[5])
        error2d2_val = np.array(log[6])

        fig, ax = plt.subplots()
        ax.plot(x, cost_train, 'r')
        ax.plot(x, cost_val, 'b')
        ax.set(xlabel='epoch', ylabel='cost', title='cost')
        plt.legend(('cost_train', 'cost_val'))
        ax.grid()
        fig.savefig(os.path.join(opt.save_dir, 'cost.png'))

        fig, ax = plt.subplots()
        ax.plot(x, error2d1_train, 'r')
        ax.plot(x, error2d2_train, 'm')
        ax.plot(x, error2d1_val, 'b')
        ax.plot(x, error2d2_val, 'c')
        ax.set(xlabel='epoch', ylabel='error2d', title='2D error (pixel)')
        plt.legend(('error2d1_train', 'error2d2_train', 'error2d1_val', 'error2d2_val'))
        ax.grid()
        fig.savefig(os.path.join(opt.save_dir, 'error2d.png'))

    #--------------------------------------------------------------------
    # test loader for final prediction
    if opt.dataset_test == 'h36m':
        test_loader = torch.utils.data.DataLoader(
            H36M17(opt.protocol, 'val'),
            batch_size = opt.batch_size,
            shuffle = False,
            num_workers = int(conf.num_threads))
    elif opt.dataset_test == 'inf':
        test_loader = torch.utils.data.DataLoader(
            MPIINF('val'),
            batch_size = opt.batch_size,
            shuffle = False,
            num_workers = int(conf.num_threads))

    # generate final prediction
    with torch.no_grad():
        test(opt.num_epochs, opt, test_loader, model)

    # to test on training set
    if opt.save_results_train == True:
        test_train_loader = torch.utils.data.DataLoader(
            H36M17(opt.protocol, 'test_train', dense=True),
            batch_size = opt.batch_size,
            shuffle = False,
            num_workers = int(conf.num_threads))

        with torch.no_grad():
            test_train(opt.num_epochs, opt, test_train_loader, model)

if __name__ == '__main__':
    main()

