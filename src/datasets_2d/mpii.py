import torch.utils.data as data
from h5py import File
import conf
import numpy as np
import cv2
from utils.utils import rnd, flip, shuffle_lr
from utils.img import transform, crop, draw_gaussian
import pdb

class MPII(data.Dataset):
    def __init__(self):
        print('==> Initializing MPII data')
        annot = {}
        tags = ['imgname', 'part', 'center', 'scale']
        f1 = File('%s/mpii/annot/%s.h5' % (conf.data_dir, 'train'), 'r')
        f2 = File('%s/mpii/annot/%s.h5' % (conf.data_dir, 'val'), 'r')
        for tag in tags:
            annot[tag] = np.concatenate((np.asarray(f1[tag]).copy(), np.asarray(f2[tag]).copy()), axis=0)
        f1.close()
        f2.close()

        self.annot = annot
        self.num_samples = len(self.annot['scale'])
        
        print('Load %d MPII samples' % (len(annot['scale'])))
    
    def load_image(self, index):
        imgname = '%s/%s' % (conf.mpii_img_dir, self.annot['imgname'][index].decode('UTF-8'))
        img = cv2.imread(imgname)
        return img
    
    def get_part_info(self, index):
        pts = self.annot['part'][index].copy()
        c = self.annot['center'][index].copy()
        s = self.annot['scale'][index]
        s = s * 200
        return pts, c, s
    
    def __getitem__(self, index):
        # get global constants
        num_joints = conf.num_joints
        res_in = conf.res_in
        res_out = conf.res_out
        res_ratio = res_in / res_out

        # get raw data
        img = self.load_image(index)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pts, c, s = self.get_part_info(index)
        r = 0

        # data augmentation (scaling and rotation)
        s = s * (2 ** rnd(conf.scale))
        r = 0 if np.random.random() < 0.6 else rnd(conf.rotate)
        inp = crop(img, c, s, r, res_in) / 255.

        # initialize valid joints
        valid2d = np.zeros((num_joints), dtype=np.float32)

        # set output heatmap and 2d pose
        pose2d = np.zeros((num_joints, 2), dtype=np.float32)
        hmap = np.zeros((int(num_joints), int(res_out), int(res_out)), dtype=np.float32)
        for i in range(16):
            if (conf.inds[i] != 7 and pts[i][0] > 1): # check whether there is a ground-truth annotation
                pt = transform(pts[i], c, s, r, res_in)
                pt = pt.astype(np.float32)
                if (pt[0] >= 0) and (pt[0] <= res_in-1) and (pt[1] >= 0) and (pt[1] <= res_in-1):
                    pose2d[conf.inds[i]] = pt
                    valid2d[conf.inds[i]] = 1.0
                    hmap[conf.inds[i]] = draw_gaussian(hmap[conf.inds[i]], pt/res_ratio+0.5, conf.std)
    
        # data augmentation (flipping and jittering)
        if np.random.random() < 0.5:
            inp = flip(inp)
            hmap = shuffle_lr(flip(hmap))
            pose2d = shuffle_lr(pose2d)
            for i in range(num_joints):
                if pose2d[i][0] > 1:
                    pose2d[i][0] = res_in - pose2d[i][0] - 1
        inp[0] = np.clip(inp[0] * (np.random.random() * .4 + .6), 0, 1)
        inp[1] = np.clip(inp[1] * (np.random.random() * .4 + .6), 0, 1)
        inp[2] = np.clip(inp[2] * (np.random.random() * .4 + .6), 0, 1)
        
        # 3d and camera information
        valid3d = 0.0
        bbox = np.array([0.0, 0.0, 255.0, 255.0], dtype=np.int32)
        pose3d = np.zeros((num_joints-1, 3), dtype=np.float32)
        cam_f = np.array([1.0, 1.0], dtype=np.float32)
        cam_c = np.array([0.0, 0.0], dtype=np.float32)
        meta3d = np.zeros((num_joints, 3), dtype=np.float32)
        action = 0
        coords_root = np.zeros((3), dtype=np.float32)
        depth_root = 0.
        depth_root_canonical = 0.

        # set data
        data = {'inp': inp, 'bbox': bbox,
            'hmap': hmap,
            'pose2d': pose2d, 'valid2d': valid2d,
            'pose3d': pose3d, 'valid3d': valid3d,
            'cam_f': cam_f, 'cam_c': cam_c,
            'meta3d': meta3d, 'action': action,
            'coords_root': coords_root,
            'depth_root': depth_root,
            'depth_root_canonical': depth_root_canonical
        }

        # return input image, output heatmap and 2d pose
        return index, data
    
    def __len__(self):
        return self.num_samples


