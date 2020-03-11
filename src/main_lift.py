import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import pdb

import conf
from opts_lift import Opts
from datasets_lift.h36m17 import H36M17
from datasets_lift.mpiinf import MPIINF
from datasets_lift.h36m17_mpiinf import H36M17_MPIINF
from models.resnet import ResNet
from train_lift import train, val, test
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

    # dataset loader (train)
    if opt.dataset_train == 'h36m':
        train_loader = torch.utils.data.DataLoader(
            H36M17(opt.protocol, 'train', True, opt.scale, opt.noise, opt.std_train, opt.std_test, opt.noise_path),
            batch_size = opt.batch_size,
            shuffle = True,
            num_workers = int(conf.num_threads))
    elif opt.dataset_train == 'inf':
        train_loader = torch.utils.data.DataLoader(
            MPIINF('train', opt.noise, opt.std_train, opt.std_test, opt.noise_path),
            batch_size = opt.batch_size,
            shuffle = True,
            num_workers = int(conf.num_threads))
    elif opt.dataset_train == 'h36m_inf':
        train_loader = torch.utils.data.DataLoader(
            H36M17_MPIINF('train', opt),
            batch_size = opt.batch_size,
            shuffle = True,
            num_workers = int(conf.num_threads))
    else:
        raise ValueError('unsupported dataset %s' % opt.dataset_train)

    # dataset loader (valid)
    if opt.dataset_test == 'h36m':
        val_loader = torch.utils.data.DataLoader(
            H36M17(opt.protocol, 'val', False, False, opt.noise, opt.std_train, opt.std_test),
            batch_size = opt.batch_size,
            shuffle = False,
            num_workers = int(conf.num_threads))
    elif opt.dataset_test == 'inf':
        val_loader = torch.utils.data.DataLoader(
            MPIINF('val', opt.noise, opt.std_train, opt.std_test),
            batch_size = opt.batch_size,
            shuffle = False,
            num_workers = int(conf.num_threads))
    else:
        raise ValueError('unsupported dataset %s' % opt.dataset_test)

    # model
    if opt.network == 'resnet':
        model = ResNet(opt.mode, conf.num_joints, opt.num_layers, opt.num_features).cuda()
    else:
        raise ValueError('unsupported model %s' % opt.network)

    # multi-gpu
    if opt.multi_gpu == True:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
    else:
        model = torch.nn.DataParallel(model, device_ids=[0])

    # optimizer
    if opt.opt_method == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr = opt.lr)
    elif opt.opt_method == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr)
    else:
        raise ValueError('unsupported optimizer %s' % opt.opt_method)
    
    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300], gamma=0.1)

    # log
    log = []
    log.append([]) # epoch
    log.append([]) # cost (train)
    log.append([]) # error3d1 (train)
    log.append([]) # error3d2 (train)
    log.append([]) # cost (val)
    log.append([]) # error3d1 (val)
    log.append([]) # error3d2 (val)

    # load model
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
        logger = Logger(opt.save_dir + '/logs')
    else:
        logger = Logger(opt.save_dir + '/logs', reset=False)

    # train
    epoch = idx_start+1
    for epoch in range(idx_start+1, opt.num_epochs+1):
        # for repeatable experiments
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed(epoch)

        # do scheduler
        scheduler.step()

        # perform one epoch of training
        cost_train, error3d1_train, error3d2_train = train(epoch, opt, train_loader, model, optimizer)
        logger.scalar_summary('cost_train', cost_train, epoch)
        logger.scalar_summary('error3d1_train', error3d1_train, epoch)
        logger.scalar_summary('error3d2_train', error3d2_train, epoch)

        # perform one epoch of validation
        with torch.no_grad():
            cost_val, error3d1_val, error3d2_val = val(epoch, opt, val_loader, model)
        logger.scalar_summary('cost_val', cost_val, epoch)
        logger.scalar_summary('error3d1_val', error3d1_val, epoch)
        logger.scalar_summary('error3d2_val', error3d2_val, epoch)

        # print message to log file
        logger.write('%d %1.1e | %.4f %.4f %.4f | %.4f %.4f %.4f\n' %
            (epoch, optimizer.param_groups[0]['lr'], cost_train, error3d1_train, error3d2_train, cost_val, error3d1_val, error3d2_val))

        #
        log[0].append(epoch)
        log[1].append(cost_train)
        log[2].append(error3d1_train)
        log[3].append(error3d2_train)
        log[4].append(cost_val)
        log[5].append(error3d1_val)
        log[6].append(error3d2_val)

        # save model
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        if epoch % opt.save_intervals == 0:
            torch.save(state, os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)))
            log_name = os.path.join(opt.save_dir, 'log_{}.pkl'.format(epoch))
            with open(log_name, 'wb') as fout:
                pickle.dump(log, fout)
    
    logger.close()

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
    error3d1_train = np.array(log[2])
    error3d2_train = np.array(log[3])
    cost_val = np.array(log[4])
    error3d1_val = np.array(log[5])
    error3d2_val = np.array(log[6])

    fig, ax = plt.subplots()
    ax.plot(x, cost_train, 'r')
    ax.plot(x, cost_val, 'b')
    ax.set(xlabel='epoch', ylabel='cost', title='cost')
    plt.legend(('cost_train', 'cost_val'))
    ax.grid()
    fig.savefig(os.path.join(opt.save_dir, 'cost.png'))

    fig, ax = plt.subplots()
    ax.plot(x, error3d1_train, 'r')
    ax.plot(x, error3d2_train, 'm')
    ax.plot(x, error3d1_val, 'b')
    ax.plot(x, error3d2_val, 'c')
    ax.set(xlabel='epoch', ylabel='error3d', title='3D error (mm)')
    plt.legend(('error3d1_train', 'error3d2_train', 'error3d1_val', 'error3d2_val'))
    ax.grid()
    fig.savefig(os.path.join(opt.save_dir, 'error3d.png'))

    #---------------------------------------------------------------------------
    # dataset loader (test)
    if opt.dataset_test == 'h36m':
        test_loader = torch.utils.data.DataLoader(
            H36M17(opt.protocol, 'test', True, False, opt.noise, opt.std_train, opt.std_test),
            batch_size = opt.batch_size,
            shuffle = False,
            num_workers = int(conf.num_threads))
    elif opt.dataset_test == 'inf':
        test_loader = torch.utils.data.DataLoader(
            MPIINF('val', opt.noise, opt.std_train, opt.std_test),
            batch_size = opt.batch_size,
            shuffle = False,
            num_workers = int(conf.num_threads))
    else:
        raise ValueError('unsupported dataset %s' % opt.dataset_test)

    # final evaluation
    with torch.no_grad():
        cost_final, error3d1_final, error3d2_final = test(epoch, opt, test_loader, model)

if __name__ == '__main__':
    main()

