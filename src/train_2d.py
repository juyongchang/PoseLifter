import os
import torch
import torch.nn as nn
import conf
from utils.utils import AverageMeter
from utils.eval import compute_error, compute_error_direct
from utils.eval import compute_error3d, compute_error3d_pa, compute_error_root
from progress.bar import Bar
import pdb

import scipy.io as sio

def weighted_mse_loss(prediction, target, weight=1.0):
    return torch.sum(weight*(prediction-target)**2)/prediction.shape[0]

def weighted_l1_loss(prediction, target, weight=1.0):
    return torch.sum(weight*torch.abs(prediction-target))/prediction.shape[0]

def step(split, epoch, opt, dataLoader, model, optimizer = None):
    # training mode
    if split == 'train':
        model.train()
    else:
        model.eval()

    # save final predictions
    if split == 'test' and opt.save_results == True:
        pred2d = []
        gt2d = []
        pred3d = []
        gt3d = []
    if split == 'test_train':
        pred = []
        gt = []

    # initialize evaluations
    cost = AverageMeter()
    error2d = []
    for i in range(2): error2d.append(AverageMeter())
    error3d = AverageMeter()
    error3d_pa = AverageMeter()
    error3d_root = AverageMeter()
    if split == 'test':
        error3dx, error3dy, error3dz = AverageMeter(), AverageMeter(), AverageMeter()
        error_action = []
        for i in range(conf.num_actions): error_action.append(AverageMeter())
    
    num_iters = len(dataLoader)
    bar = Bar('==>', max=num_iters)
    
    # for each mini-batch
    for i, (idx, data) in enumerate(dataLoader):
        img = data['inp'].to("cuda")
        hmap = data['hmap'].to("cuda")
        pose2d = data['pose2d'].to("cuda")
        valid2d = data['valid2d'].to("cuda")
        valid2d_sum = valid2d.sum().item()

        # for 3d
        pose3d = data['pose3d'].to("cuda")
        valid3d = data['valid3d'].float().to("cuda")
        valid3d_sum = valid3d.sum().item()
        bbox = data['bbox'].float().to("cuda")

        # for depth
        cam_f = data['cam_f'].to("cuda")
        cam_c = data['cam_c'].to("cuda")
        meta3d = data['meta3d'].to("cuda")
        action = data['action'].to("cuda")
        coords_root = data['coords_root'].to("cuda")
        depth_root = data['depth_root'].to("cuda")
        depth_root_canonical = data['depth_root_canonical'].to("cuda")

        # constants
        nb = img.shape[0]
        nj = conf.num_joints

        # forward propagation
        if split == 'test':
            model.module.set_fliptest(opt.fliptest)
        if opt.lift3d == 0:
            outputs = model(img)
        else:
            outputs = model([img, bbox, cam_c])

        # compute cost
        loss = 0.0
        if opt.lift3d == 0:
            if opt.integral == 0:
                loss += weighted_mse_loss(outputs[0], hmap, valid2d.reshape(nb, nj, 1, 1))
            else:
                loss += weighted_mse_loss(outputs[0], hmap, valid2d.reshape(nb, nj, 1, 1))
                loss += weighted_l1_loss(outputs[1], pose2d, valid2d.reshape(nb, nj, 1))*opt.weight1
        else:
            loss += weighted_l1_loss(outputs[2], pose3d, valid3d.reshape(nb, 1, 1))

        # update model parameters with backpropagation
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update evaluations
        cost.update(loss.detach().item(), nb)
        if opt.lift3d == 0:
            if opt.integral == 0:
                error2d[0].update(compute_error(outputs[0].detach(), pose2d.detach(), valid2d.detach()), valid2d_sum)
            else:
                error2d[0].update(compute_error(outputs[0].detach(), pose2d.detach(), valid2d.detach()), valid2d_sum)
                error2d[1].update(compute_error_direct(outputs[1].detach(), pose2d.detach(), valid2d.detach()), valid2d_sum)
        else:
            # unnormalized 2d pose
            pred = outputs[1].detach()
            pred = pred / 255.0
            pred = pred * torch.reshape(bbox[:, 2:4], (nb, 1, 2))
            pred = pred + torch.reshape(bbox[:, 0:2], (nb, 1, 2))

            # get predicted root coordinates
            x = pred[:, conf.root, 0]
            y = pred[:, conf.root, 1]
            cx = cam_c[:, 0].detach()
            cy = cam_c[:, 1].detach()
            if opt.lift3d == 1: # Z is not canonical depth
                Z = outputs[3].detach()
                f = torch.sqrt(torch.prod(cam_f.detach(), 1))
                X = (x-cx)*Z/f
                Y = (y-cy)*Z/f
                pred_root = torch.cat((X.view(nb,1), Y.view(nb,1), Z.view(nb,1)), 1)
            elif opt.lift3d == 2: # Z is canonical depth
                Z = outputs[3].detach()
                X = (x-cx)*Z/conf.f0
                Y = (y-cy)*Z/conf.f0
                f = torch.sqrt(torch.prod(cam_f.detach(), 1))/conf.f0
                Z = Z*f
                pred_root = torch.cat((X.view(nb,1), Y.view(nb,1), Z.view(nb,1)), 1)

            # evaluate 2d estimate
            error2d[0].update(compute_error(outputs[0].detach(), pose2d.detach(), valid2d.detach()), valid2d_sum)
            error2d[1].update(compute_error_direct(outputs[1].detach(), pose2d.detach(), valid2d.detach()), valid2d_sum)

            # evaluate 3d estimate
            error3d.update(compute_error3d(outputs[2].detach(), pose3d.detach(), valid3d.detach()))
            error3d_pa.update(compute_error3d_pa(outputs[2].detach(), pose3d.detach()))

            # evaluate root coordinates (MRPE)
            error3d_root.update(compute_error_root(pred_root, coords_root.detach()))
            error3dx.update(torch.sum(torch.abs(pred_root[:,0]-coords_root[:,0].detach()))/nb)
            error3dy.update(torch.sum(torch.abs(pred_root[:,1]-coords_root[:,1].detach()))/nb)
            error3dz.update(torch.sum(torch.abs(pred_root[:,2]-coords_root[:,2].detach()))/nb)

            if split == 'test':
                for j in range(nb):
                    if opt.protocol == 1:
                        error_action[int(action[j].item())-2].update(compute_error3d(outputs[2][j].reshape(1,nj-1,3).detach(), pose3d[j].reshape(1,nj-1,3).detach(), valid3d[j].detach()))
                    elif opt.protocol == 2:
                        error_action[int(action[j].item())-2].update(compute_error3d_pa(outputs[2][j].reshape(1,nj-1,3).detach(), pose3d[j].reshape(1,nj-1,3).detach()))

        # for computing error statistics
        if split == 'test' and opt.save_results == True:
            pred2d.append(outputs[1].detach())
            gt2d.append(pose2d.detach())
            if opt.lift3d != 0:
                temp = torch.zeros(nb, nj, 3).to("cuda")
                temp[:,:conf.root,:] = outputs[2][:,:conf.root,:]
                temp[:,conf.root+1:,:] = outputs[2][:,conf.root:,:]
                #temp = temp + torch.reshape(outputs[3], (nb,1,3))
                pred3d.append(temp.detach())
                gt3d.append(meta3d)

        # for computing error statistics
        if split == 'test_train':
            pred.append(outputs[1].detach())
            gt.append(pose2d.detach())

        msg = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:}'.format(epoch, i, num_iters, split=split, total=bar.elapsed_td, eta=bar.eta_td)
        if split == 'train':
            msg = '{} | LR: {:1.1e}'.format(msg, optimizer.param_groups[0]['lr'])
        msg = '{} | Cost {cost.avg:.2f}'.format(msg, cost=cost)
        for j in range(2):
            msg = '{} | E{:d} {:.2f}'.format(msg, j, error2d[j].avg)
        msg = '{} | E3D {:.2f} | E3D_PA {:.2f}'.format(msg, error3d.avg, error3d_pa.avg)
        Bar.suffix = msg
        bar.next()

    bar.finish()

    # save final test results
    if split == 'test' and opt.save_results == True:
        pred2d = torch.cat(pred2d, dim=0)
        gt2d = torch.cat(gt2d, dim=0)
        if opt.lift3d == 0:
            result = {'pred2d': pred2d, 'gt2d': gt2d}
        else:
            pred3d = torch.cat(pred3d, dim=0)
            gt3d = torch.cat(gt3d, dim=0)
            result = {'pred2d': pred2d, 'gt2d': gt2d, 'pred3d': pred3d, 'gt3d': gt3d}
        filename = os.path.join(opt.save_dir, 'result.pth')
        torch.save(result, filename)

        if opt.lift3d == 0:
            result = {'pred2d': pred2d.cpu().numpy(), 'gt2d': gt2d.cpu().numpy()}
        else:
            result = {'pred2d': pred2d.cpu().numpy(), 'gt2d': gt2d.cpu().numpy(), 'pred3d': pred3d.cpu().numpy(), 'gt3d': gt3d.cpu().numpy()}
        filename = os.path.join(opt.save_dir, 'result.mat')
        sio.savemat(filename, result)

    # save train results
    if split == 'test_train':
        pred = torch.cat(pred, dim=0)
        gt = torch.cat(gt, dim=0)
        result = {'pred': pred, 'gt': gt}
        filename = os.path.join(opt.save_dir, 'test_train.pth')
        torch.save(result, filename)

    #
    if split == 'test':
        if opt.fliptest == True:
            file = open(os.path.join(opt.save_dir, 'result_fliptest.txt'), 'w')
        else:
            file = open(os.path.join(opt.save_dir, 'result.txt'), 'w')
        file.write('Loss for test set = {:6f}\n'.format(cost.avg))
        file.write('(Heatmap) 2D pixel error for test set = {:6f}\n'.format(error2d[0].avg))
        file.write('(Coord) 2D pixel error for test set = {:6f}\n'.format(error2d[1].avg))
        file.write('---------------------------------------------------\n')
        file.write('3D MPJPE error for test set = {:6f}\n'.format(error3d.avg))
        file.write('3D PA MPJPE error for test set = {:6f}\n'.format(error3d_pa.avg))
        file.write('Global root error for test set = {:6f}\n'.format(error3d_root.avg))
        file.write('---------------------------------------------------\n')
        file.write('Root error in X direction for test set = {:6f}\n'.format(error3dx.avg))
        file.write('Root error in Y direction for test set = {:6f}\n'.format(error3dy.avg))
        file.write('Root error in Z direction for test set = {:6f}\n'.format(error3dz.avg))
        file.write('---------------------------------------------------\n')
        for i in range(conf.num_actions):
            file.write('3D MPJPE error for action %d = %.6f\n' % (i, error_action[i].avg))
        file.close()

    return cost.avg, [error2d[0].avg, error2d[1].avg], error3d.avg, error3d_pa.avg

def train(epoch, opt, train_loader, model, optimizer):
    return step('train', epoch, opt, train_loader, model, optimizer)

def val(epoch, opt, val_loader, model):
    return step('val', epoch, opt, val_loader, model)

def test(epoch, opt, test_loader, model):
    return step('test', epoch, opt, test_loader, model)

def test_train(epoch, opt, test_train_loader, model):
    return step('test_train', epoch, opt, test_train_loader, model)

