import os
import torch
import torch.nn as nn
import torchvision.models as models
from models.resnet2d import ResNet
from collections import OrderedDict
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

class ResNetInt(nn.Module):
    def __init__(self, network, num_joints, model_path=None):
        super(ResNetInt, self).__init__()

        # heatmap regression module
        self.resnet = ResNet(network, num_joints)

        # load pretrained heatmap regression model
        if model_path is not None:
            if os.path.isfile(model_path):
                print('Load pretrained heatmap regression model..')
                state = torch.load(model_path)
                pretrained_dict = state['model']
                model_dict = self.resnet.state_dict()
                new_pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
                model_dict.update(new_pretrained_dict)
                self.resnet.load_state_dict(model_dict)
            else:
                raise ValueError('model does not exist: %s' % model_path)

        # for integration
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('wx', torch.arange(64.0)*4.0+2.0)
        self.wx = self.wx.reshape(1,64).repeat(64,1).reshape(64*64,1)
        self.register_buffer('wy', torch.arange(64.0)*4.0+2.0)
        self.wy = self.wy.reshape(64,1).repeat(1,64).reshape(64*64,1)

        self.network = network
        self.num_joints = num_joints
        self.fliptest = False

    def forward(self, img):
        H = self.resnet(img)[0]

        if self.fliptest == True:
            img_flipped = torch.flip(img, [3])
            H_flipped = self.resnet(img_flipped)[0]
            H1 = torch.flip(H_flipped, [3])
            index = torch.tensor(flip_index).to(H1.device)
            H2 = H1.clone()
            H2.index_copy_(1, index, H1)
            H = (H + H2) * 0.5

        hmap = H.clone()
        num_batch = hmap.shape[0]
        hmap = torch.reshape(hmap, (num_batch, self.num_joints, 64*64))
        hmap = self.relu(hmap) # for numerical stability
        denom = torch.sum(hmap, 2, keepdim=True).add_(1e-6) # for numerical stability
        hmap = hmap / denom

        x = torch.matmul(hmap, self.wx)
        y = torch.matmul(hmap, self.wy)
        coord = torch.cat((x,y), 2)

        return [H, coord]

    def set_fliptest(self, val):
        self.fliptest = val

