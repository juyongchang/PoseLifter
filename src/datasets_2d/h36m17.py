import torch.utils.data as data
from h5py import File
import conf
import numpy as np
import cv2
from utils.utils import rnd, flip, shuffle_lr
from utils.img import transform, crop, draw_gaussian
import pdb

subject_list = [[[1, 5, 6, 7], [8]], [[1, 5, 6, 7, 8], [9, 11]], [[1, 5, 6, 7, 8, 9], [11]]]
flip_index = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]

class H36M17(data.Dataset):
    def __init__(self, protocol, split, dense=False):
        print('==> Initializing H36M %s data' % (split))
        annot = {}
        tags = ['idx', 'pose2d', 'pose3d', 'bbox', 'cam_f', 'cam_c', 'subject', 'action', 'subaction', 'camera']
        f = File('%s/h36m/h36m17.h5' % (conf.data_dir), 'r')
        for tag in tags:
            annot[tag] = np.asarray(f[tag]).copy()
        f.close()

        if dense == False:
            idxs = np.mod(annot['idx'], 50) == 1
            idxs = np.arange(annot['idx'].shape[0])[idxs]
            for tag in tags:
                annot[tag] = annot[tag][idxs]

        idxs = np.full(annot['idx'].shape[0], False)
        subject = subject_list[protocol][1-int(split=='train' or split=='test_train')]
        for i in range(len(subject)):
            idxs = idxs + (annot['subject']==subject[i])
        idxs = np.arange(annot['idx'].shape[0])[idxs]
        for tag in tags:
            annot[tag] = annot[tag][idxs]

        self.protocol = protocol
        self.split = split
        self.dense = dense
        self.annot = annot
        self.num_samples = len(self.annot['idx'])

        print('Load %d H36M %s samples' % (self.num_samples, self.split))

    def load_image(self, index):
        dirname = 's_%02d_act_%02d_subact_%02d_ca_%02d' % (self.annot['subject'][index], \
            self.annot['action'][index], self.annot['subaction'][index], self.annot['camera'][index])
        imgname = '%s/%s/%s_%06d.jpg' % (conf.h36m_img_dir, dirname, dirname, self.annot['idx'][index])
        img = cv2.imread(imgname)
        return img

    def get_part_info(self, index):
        pose2d = self.annot['pose2d'][index].copy()
        bbox = self.annot['bbox'][index].copy()
        pose3d = self.annot['pose3d'][index].copy()
        cam_f = self.annot['cam_f'][index].copy()
        cam_c = self.annot['cam_c'][index].copy()
        action = self.annot['action'][index].copy()
        return pose2d, bbox, pose3d, cam_f, cam_c, action

    def __getitem__(self, index):
        if self.split == 'train':
            index = np.random.randint(self.num_samples)

        # get global constants
        num_joints = conf.num_joints
        res_in = conf.res_in
        res_out = conf.res_out
        res_ratio = res_in / res_out

        # get image
        img = self.load_image(index)

        # get 2d/3d pose and bounding box
        pts, bbox, meta3d, cam_f, cam_c, action = self.get_part_info(index)
        cam_f = cam_f.astype(np.float32)
        cam_c = cam_c.astype(np.float32)
        action = action.item()

        # set 2d pose
        pts = pts - bbox[0:2]
        pts = pts / bbox[2:4]
        pts = pts * float(res_in - 1)

        # set 3d pose
        pose3d = meta3d.copy()
        pose3d = pose3d - pose3d[conf.root]

        # data augmentation (small random translation)
        inp = np.zeros_like(img)
        if self.split == 'train':
            xr = np.random.randint(9)-4
            yr = np.random.randint(9)-4
            in_x1, in_x2 = max(0,-xr), min(256,256-xr)
            in_y1, in_y2 = max(0,-yr), min(256,256-yr)
            out_x1, out_x2 = max(0,xr), min(256,256+xr)
            out_y1, out_y2 = max(0,yr), min(256,256+yr)
            inp[out_y1:out_y2, out_x1:out_x2, :] = img[in_y1:in_y2, in_x1:in_x2, :]
            pts[:,0] = pts[:,0] + xr
            pts[:,1] = pts[:,1] + yr
        else:
            inp[:,:,:] = img[:,:,:]

        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp = inp.transpose(2,0,1).astype(np.float32)
        
        # normalization
        inp = inp / 255.0

        # set valid joints
        valid2d = np.ones((num_joints), dtype=np.float32)
        valid3d = 1.0

        # set output heatmap and 2d pose
        pose2d = np.zeros((num_joints, 2), dtype=np.float32)
        hmap = np.zeros((int(num_joints), int(res_out), int(res_out)), dtype=np.float32)
        for i in range(num_joints):
            pt = pts[i].astype(np.float32)
            pose2d[i] = pt
            hmap[i] = draw_gaussian(hmap[i], pt/res_ratio+0.5, conf.std)

        # data augmentation (flipping and color jittering)
        if self.split == 'train':
            if np.random.random() < 0.5:
                inp = flip(inp)
                hmap = flip(hmap)
                hmap_flip = hmap.copy()
                for i in range(len(flip_index)):
                    hmap_flip[i] = hmap[flip_index[i]].copy()
                hmap = hmap_flip.copy()
                pose2d_flip = pose2d.copy()
                for i in range(len(flip_index)):
                    pose2d_flip[i] = pose2d[flip_index[i]].copy()
                pose2d = pose2d_flip.copy()
                pose2d[:, 0] = conf.res_in - pose2d[:, 0]
                pose3d_flip = pose3d.copy()
                for i in range(len(flip_index)):
                    pose3d_flip[i] = pose3d[flip_index[i]].copy()
                pose3d = pose3d_flip.copy()
                pose3d[:, 0] *= -1
            inp[0] = np.clip(inp[0] * (np.random.random() * .4 + .6), 0, 1)
            inp[1] = np.clip(inp[1] * (np.random.random() * .4 + .6), 0, 1)
            inp[2] = np.clip(inp[2] * (np.random.random() * .4 + .6), 0, 1)

        # root coordinates
        coords_root = meta3d[conf.root].copy()
        depth_root = coords_root[2].copy()
        depth_root_canonical = coords_root[2].copy() / np.sqrt(np.prod(cam_f))

        # set 3d pose
        pose3d = np.delete(pose3d, (conf.root), axis=0)

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

        # return
        return index, data

    def __len__(self):
        return self.num_samples

