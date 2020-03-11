import os
import torch
import torch.nn as nn
from models.resnet2d_int import ResNetInt
from models.resnet import ResNet
import pdb

class ResNetLift(nn.Module):
    def __init__(self, network, num_joints, num_layers, num_features, mode, model_2d_path, model_lift_path):
        super(ResNetLift, self).__init__()

        # 2d pose estimation module
        self.model_2d = ResNetInt(network, num_joints)
        if model_2d_path is not None:
            if os.path.isfile(model_2d_path):
                print('Load pretrained 2D pose estimation model..')
                state = torch.load(model_2d_path)
                pretrained_dict = state['model']
                model_dict = self.model_2d.state_dict()
                new_pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
                model_dict.update(new_pretrained_dict)
                self.model_2d.load_state_dict(model_dict)
            else:
                raise ValueError('model does not exist: %s' % model_2d_path)

        # 2d-to-3d pose lifting module
        self.model_lift = ResNet(mode, num_joints, num_layers, num_features)
        if model_lift_path is not None:
            if os.path.isfile(model_lift_path):
                print('Load pretrained 2D pose estimation model..')
                state = torch.load(model_lift_path)
                pretrained_dict = state['model']
                model_dict = self.model_lift.state_dict()
                new_pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
                model_dict.update(new_pretrained_dict)
                self.model_lift.load_state_dict(model_dict)
            else:
                raise ValueError('model does not exist: %s' % model_lift_path)

    def forward(self, inp):
        [img, bbox, cam_c] = inp

        # 2d prediction
        [H, pred2d] = self.model_2d(img)

        # 3d prediction
        pose2d = pred2d.clone()
        [pose_local, depth_root] = self.model_lift([pose2d, bbox, cam_c])

        return [H, pred2d, pose_local, depth_root]

    def set_fliptest(self, val):
        self.model_2d.set_fliptest(val)
        self.model_lift.set_fliptest(val)

