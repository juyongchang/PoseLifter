import os
import numpy as np
import torch
from utils.utils import AverageMeter
from utils.eval import compute_error3d, compute_error3d_pa, compute_error_root
from utils.eval import compute_error3d_x, compute_error3d_y, compute_error3d_z
from progress.bar import Bar
import pdb
import conf

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

    #
    if opt.analysis == True:
        std2d = []
        depth = []

    # initialize evaluations
    cost, error3d1, error3d2 = AverageMeter(), AverageMeter(), AverageMeter()
    error3d3 = AverageMeter()
    error3dx, error3dy, error3dz = AverageMeter(), AverageMeter(), AverageMeter()
    
    num_iters = len(dataLoader)
    bar = Bar('==>', max=num_iters)
    
    # for each mini-batch,
    for i, (data) in enumerate(dataLoader):
        pose2d = data['pose2d'].float().to("cuda")
        bbox = data['bbox'].float().to("cuda")
        pose3d = data['pose3d'].float().to("cuda")
        coords_root = data['coords_root'].float().to("cuda")
        depth_root = data['depth_root'].float().to("cuda")
        depth_root_canonical = data['depth_root_canonical'].float().to("cuda")
        cam_f = data['cam_f'].float().to("cuda")
        cam_c = data['cam_c'].float().to("cuda")
        meta2d = data['meta2d'].float().to("cuda")

        # forward propagation
        if split == 'test':
            model.module.fliptest = opt.fliptest
        outputs = model([pose2d, bbox, cam_c])

        #
        nb = pose2d.size(0)
        nj = conf.num_joints

        # compute cost
        loss = 0
        loss += weighted_l1_loss(outputs[0], pose3d)
        if opt.canonical == False:
            loss += weighted_l1_loss(outputs[1], depth_root) * opt.weight_root
        else:
            loss += weighted_l1_loss(outputs[1], depth_root_canonical) * opt.weight_root

        # update model parameters with backpropagation
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # unnormalized 2d pose
        pose2d = pose2d.detach()
        pose2d = pose2d / 255.0
        pose2d = pose2d * torch.reshape(bbox[:, 2:4], (nb, 1, 2))
        pose2d = pose2d + torch.reshape(bbox[:, 0:2], (nb, 1, 2))

        # get predicted root coordinates
        x = pose2d[:, conf.root, 0]
        y = pose2d[:, conf.root, 1]
        cx = cam_c[:, 0].detach()
        cy = cam_c[:, 1].detach()
        if opt.canonical == False:
            Z = outputs[1].detach()
            f = torch.sqrt(torch.prod(cam_f.detach(), 1))
            X = (x-cx)*Z/f
            Y = (y-cy)*Z/f
            pred_root = torch.cat((X.view(nb,1), Y.view(nb,1), Z.view(nb,1)), 1)
        else:
            Z = outputs[1].detach()
            X = (x-cx)*Z/conf.f0
            Y = (y-cy)*Z/conf.f0
            f = torch.sqrt(torch.prod(cam_f.detach(), 1))/conf.f0
            Z = Z*f
            pred_root = torch.cat((X.view(nb,1), Y.view(nb,1), Z.view(nb,1)), 1)

        # update evaluations
        cost.update(loss.detach().item(), pose2d.size(0))
        error3d1.update(compute_error3d(outputs[0].detach(), pose3d.detach()))
        error3d2.update(compute_error_root(pred_root, coords_root.detach()))
        if split == 'test':
            error3d3.update(compute_error3d_pa(outputs[0].detach(), pose3d.detach()))
            error3dx.update(compute_error3d_x(outputs[0].detach(), pose3d.detach()))
            error3dy.update(compute_error3d_y(outputs[0].detach(), pose3d.detach()))
            error3dz.update(compute_error3d_z(outputs[0].detach(), pose3d.detach()))

        #
        if opt.analysis == True:
            std2d.append(outputs[2].detach())
            depth.append(outputs[1].detach())

        Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Cost {cost.avg:.6f} | Error3D1 {error3d1.avg:.6f} | Error3D2 {error3d2.avg:.6f}'.format(epoch, i, num_iters, split=split, total=bar.elapsed_td, eta=bar.eta_td, cost=cost, error3d1=error3d1, error3d2=error3d2)
        bar.next()
    
    bar.finish()

    if split == 'test':
        if (opt.std_test > 0.0) and (opt.std_test < 1.0):
            file = open(os.path.join(opt.save_dir, 'result_noise_gaussian%.3f.txt' % (opt.std_test)), 'w')
        elif (opt.std_test > 1.0) and (opt.std_test < 2.0):
            file = open(os.path.join(opt.save_dir, 'result_noise_uniform%.3f.txt' % (opt.std_test)), 'w')
        else:
            if opt.fliptest == True:
                file = open(os.path.join(opt.save_dir, 'result_fliptest.txt'), 'w')
            else:
                file = open(os.path.join(opt.save_dir, 'result.txt'), 'w')
        file.write('L1 loss for test set = {:6f}\n'.format(cost.avg))
        file.write('3D MPJPE error for test set = {:6f}\n'.format(error3d1.avg))
        file.write('3D Root error for test set = {:6f}\n'.format(error3d2.avg))
        file.write('3D PA MPJPE error for test set = {:6f}\n'.format(error3d3.avg))
        file.write('3D error in X axis = {:6f}\n'.format(error3dx.avg))
        file.write('3D error in Y axis = {:6f}\n'.format(error3dy.avg))
        file.write('3D error in Z axis = {:6f}\n'.format(error3dz.avg))
        file.close()

        if opt.analysis == True:
            std2d = torch.cat(std2d, dim=0).squeeze().cpu().numpy() * 1000.
            depth = torch.cat(depth, dim=0).squeeze().cpu().numpy() / 1000.

            filename = os.path.join(opt.save_dir, 'result.npz')
            np.savez(filename, std2d, depth)

    return cost.avg, error3d1.avg, error3d2.avg

def train(epoch, opt, train_loader, model, optimizer):
    return step('train', epoch, opt, train_loader, model, optimizer)

def val(epoch, opt, val_loader, model):
    return step('val', epoch, opt, val_loader, model)

def test(epoch, opt, test_loader, model):
    return step('test', epoch, opt, test_loader, model)

