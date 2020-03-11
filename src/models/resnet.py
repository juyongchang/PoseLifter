import torch
import torch.nn as nn
import conf
import pdb

flip_index = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]
flip_index_ = [3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 13, 14, 15, 10, 11, 12]

class ResNetModule(nn.Module):
    def __init__(self, num_features):
        super(ResNetModule, self).__init__()

        modules = []
        modules.append(nn.BatchNorm1d(num_features))
        modules.append(nn.Dropout(0.5))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(num_features, num_features))
        modules.append(nn.BatchNorm1d(num_features))
        modules.append(nn.Dropout(0.5))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(num_features, num_features))

        # set weights
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.submod = nn.Sequential(*modules)

    def forward(self, x):
        return self.submod(x) + x

class ResNet(nn.Module):
    def __init__(self, mode, num_joints, num_layers, num_features):
        super(ResNet, self).__init__()

        self.mode = mode
        self.num_joints = num_joints
        if mode == 0 or mode == 2:
            self.num_in = 2*num_joints
        elif mode == 1:
            self.num_in = 2*num_joints+3
        elif mode == 3:
            self.num_in = 2*num_joints+2
        elif mode == 4:
            self.num_in = 2*num_joints+1
        self.num_out = 3*(num_joints-1)+1
        self.num_layers = num_layers
        self.num_features = num_features

        mod = []
        mod.append(nn.Linear(self.num_in, num_features))
        for i in range(num_layers):
            mod.append(ResNetModule(num_features))
        mod.append(nn.Linear(num_features, self.num_out))

        # set weights
        nn.init.normal_(mod[0].weight, mean=0, std=0.001)
        nn.init.constant_(mod[0].bias, 0)
        nn.init.normal_(mod[-1].weight, mean=0, std=0.001)
        nn.init.constant_(mod[-1].bias, 0)

        self.mod = nn.ModuleList(mod)

        self.fliptest = False

    def forward(self, inp):
        pose2d = inp[0].clone()
        bbox = inp[1].clone()
        cam_c = inp[2].clone()

        nb = pose2d.shape[0]

        if self.mode == 0 or self.mode == 1 or self.mode == 3 or self.mode == 4:
            pose2d = pose2d / 255.0
            pose2d = pose2d * torch.reshape(bbox[:,2:4], (nb, 1, 2))
            pose2d = pose2d + torch.reshape(bbox[:,0:2], (nb, 1, 2))

            pose2d = pose2d - torch.reshape(cam_c, (nb, 1, 2))

            mean2d = torch.mean(pose2d, 1, keepdim=True)
            dist = torch.sqrt(torch.sum((pose2d-mean2d)**2.0, 2, keepdim=True))
            std2d = torch.std(dist, 1, keepdim=True)
            pose2d = (pose2d - mean2d) / std2d
            mean2d = mean2d * 0.001
            std2d = std2d * 0.001

            if self.mode == 0: # normalized 2d pose
                x = pose2d.reshape(nb, -1)
            elif self.mode == 1: # normalized 2d pose, mean, std
                x = torch.cat((pose2d.reshape(nb, -1), mean2d.reshape(nb, -1), std2d.reshape(nb, -1)), 1)
            elif self.mode == 3: # mean
                x = torch.cat((pose2d.reshape(nb, -1), mean2d.reshape(nb, -1)), 1)
            elif self.mode == 4: # std
                x = torch.cat((pose2d.reshape(nb, -1), std2d.reshape(nb, -1)), 1)
        elif self.mode == 2: # unnormalized 2d pose
            x = pose2d.reshape(nb, -1)
        else:
            raise ValueError('unsupported poselifter mode')

        x = self.mod[0](x)
        for i in range(self.num_layers):
            x = self.mod[i+1](x)
        x = self.mod[-1](x)

        pose_local = x[:, 0:(self.num_out-1)].view(-1, self.num_joints-1, 3)
        depth_root = x[:, self.num_out-1]

        if self.fliptest == True:
            [pose_local_flip, depth_root_flip] = self.forward_flip(inp)
            pose_local = (pose_local + pose_local_flip) * 0.5
            depth_root = (depth_root + depth_root_flip) * 0.5

        return [pose_local, depth_root]

    def forward_flip(self, inp):
        pose2d_orig = inp[0].clone()
        bbox = inp[1].clone()
        cam_c = inp[2].clone()

        nb = pose2d_orig.shape[0]

        index = torch.tensor(flip_index).to(pose2d_orig.device)
        pose2d = pose2d_orig.clone()
        pose2d.index_copy_(1, index, pose2d_orig)

        if self.mode == 0 or self.mode == 1 or self.mode == 3 or self.mode == 4:
            pose2d = pose2d / 255.0
            pose2d = pose2d * torch.reshape(bbox[:,2:4], (nb, 1, 2))
            pose2d = pose2d + torch.reshape(bbox[:,0:2], (nb, 1, 2))

            pose2d[:, :, 0] = conf.width - pose2d[:, :, 0]
            cam_c[:, 0] = conf.width - cam_c[:, 0]

            pose2d = pose2d - torch.reshape(cam_c, (nb, 1, 2))

            mean2d = torch.mean(pose2d, 1, keepdim=True)
            dist = torch.sqrt(torch.sum((pose2d-mean2d)**2.0, 2, keepdim=True))
            std2d = torch.std(dist, 1, keepdim=True)
            pose2d = (pose2d - mean2d) / std2d
            mean2d = mean2d * 0.001
            std2d = std2d * 0.001

            if self.mode == 0:
                x = pose2d.reshape(nb, -1)
            elif self.mode == 1:
                x = torch.cat((pose2d.reshape(nb, -1), mean2d.reshape(nb, -1), std2d.reshape(nb, -1)), 1)
            elif self.mode == 3: # mean only
                x = torch.cat((pose2d.reshape(nb, -1), mean2d.reshape(nb, -1)), 1)
            elif self.mode == 4: # std only
                x = torch.cat((pose2d.reshape(nb, -1), std2d.reshape(nb, -1)), 1)
        elif self.mode == 2:
            x = pose2d.reshape(nb, -1)
        else:
            raise ValueError('unsupported poselifter mode')

        x = self.mod[0](x)
        for i in range(self.num_layers):
            x = self.mod[i+1](x)
        x = self.mod[-1](x)

        pose_local = x[:, 0:(self.num_out-1)].view(-1, self.num_joints-1, 3)
        depth_root = x[:, self.num_out-1]

        pose_local[:, :, 0] *= -1

        index_ = torch.tensor(flip_index_).to(pose_local.device)
        pose_local_ = pose_local.clone()
        pose_local_.index_copy_(1, index_, pose_local)

        return [pose_local_, depth_root]

    def set_fliptest(self, val):
        self.fliptest = val

