import torch
import torch.nn as nn
import torchvision.models as models
import pdb

flip_index = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]

# modules[0]: conv2d (stride=2, res=128)
# modules[1]: batchnorm
# modules[2]: relu
# modules[3]: maxpool (stride=2, res=64)
# modules[4]: 1st conv block - output: C2 (256 channels)
# modules[5]: 2nd conv block (stride=2, res=32) - output: C3 (512 channels)
# modules[6]: 3rd conv block (stride=2, res=16) - output: C4 (1024 channels)
# modules[7]: 4th conv block (stride=2, res=8) - output: C5 (2048 channels)
# modules[8]: 1st deconv block (res=16)
# modules[9]: 2nd deconv block (res=32)
# modules[10]: 3rd deconv block (res=64)
# modules[11]: regression layer

class ResNet(nn.Module):
    def __init__(self, network, num_joints):
        super(ResNet, self).__init__()
        
        # load pretrained resnet model
        if network == 'resnet50':
            pretrained = models.resnet50(pretrained=True)
        elif network == 'resnet101':
            pretrained = models.resnet101(pretrained=True)
        elif network == 'resnet152':
            pretrained = models.resnet152(pretrained=True)
        elif network == 'resnext101':
            pretrained = models.resnext101_32x8d(pretrained=True)
        else:
            raise ValueError('unsupported network %s' % network)
        
        # remove last 2 layers
        modules = list(pretrained.children())[:-2]
        
        # add 1st deconv block (stride = 16)
        modules.append(nn.Sequential(nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # add 2nd deconv block (stride = 32)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # add 3rd deconv block (stride = 64)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # add regression layer
        modules.append(nn.Conv2d(256, num_joints, 1))

        self.module = nn.ModuleList(modules)

        self.network = network
        self.num_joints = num_joints
        self.fliptest = False

    def forward(self, img):
        x = self.module[0](img)
        x = self.module[1](x)
        x = self.module[2](x)
        x = self.module[3](x)
        C2 = self.module[4](x)
        C3 = self.module[5](C2)
        C4 = self.module[6](C3)
        C5 = self.module[7](C4)
        H = self.module[8](C5)
        H = self.module[9](H)
        H = self.module[10](H)
        H = self.module[11](H)

        if self.fliptest == True:
            img_flipped = torch.flip(img, [3])
            x_flipped = self.module[0](img_flipped)
            x_flipped = self.module[1](x_flipped)
            x_flipped = self.module[2](x_flipped)
            x_flipped = self.module[3](x_flipped)
            C2_flipped = self.module[4](x_flipped)
            C3_flipped = self.module[5](C2_flipped)
            C4_flipped = self.module[6](C3_flipped)
            C5_flipped = self.module[7](C4_flipped)
            H_flipped = self.module[8](C5_flipped)
            H_flipped = self.module[9](H_flipped)
            H_flipped = self.module[10](H_flipped)
            H_flipped = self.module[11](H_flipped)
            H1 = torch.flip(H_flipped, [3])
            index = torch.tensor(flip_index).to(H1.device)
            H2 = H1.clone()
            H2.index_copy_(1, index, H1)
            H = (H + H2) * 0.5

        return [H]

    def set_fliptest(self, val):
        self.fliptest = val

