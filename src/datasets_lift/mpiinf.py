import torch
import torch.utils.data as data
from h5py import File
import conf
import numpy as np
import pdb

flip_index = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]

class MPIINF(data.Dataset):
    def __init__(self, split, noise=0, std_train=0, std_test=0, noise_path=None):
        print('==> Initializing MPI_INF %s data' % (split))

        annot = {}
        tags = ['pose2d', 'pose3d', 'bbox', 'cam_f', 'cam_c', 'subject', 'sequence', 'video']
        f = File('%s/inf/inf_%s.h5' % (conf.data_dir, split), 'r')
        for tag in tags:
            annot[tag] = np.asarray(f[tag]).copy()
        f.close()

        self.split = split
        self.noise = noise
        self.std_train = std_train
        self.std_test = std_test
        self.noise_path = noise_path
        self.annot = annot
        self.num_samples = self.annot['pose2d'].shape[0]

        # image size
        self.width = 2048 * 0.5
        self.height = 2048 * 0.5

        # load error statistics
        self.load_error_stat()

        print('Load %d MPI_INF %s samples' % (self.num_samples, self.split))

    def get_part_info(self, index):
        pose2d = self.annot['pose2d'][index].copy()
        bbox = self.annot['bbox'][index].copy()
        pose3d = self.annot['pose3d'][index].copy()
        cam_f = self.annot['cam_f'][index].copy()
        cam_c = self.annot['cam_c'][index].copy()
        return pose2d, bbox, pose3d, cam_f, cam_c

    def load_error_stat(self):
        # load error stat
        if self.split == 'train':
            if self.noise == 0: # do not use noise
                pass
            elif self.noise == 1: # use specified gaussian noise
                pass
            elif self.noise == 2: # use estimated 2d pose
                filename = '%s_protocol%d/resnet152-int/fusion/rmsprop_lr1.0e-05_batch48/test_train.pth' % (conf.exp_dir, self.protocol)
                result = torch.load(filename)
                self.annot['pose2d'] = result['pred'].cpu().numpy()
            elif self.noise == 3: # use estimated single gaussian noise
                filename = '%s/%s' % (conf.exp_dir, self.noise_path)
                result = torch.load(filename)
                mean = result['mean'].numpy() / float(conf.res_in - 1)
                self.mean = mean[0]
                std = result['std'].numpy() / float(conf.res_in - 1)
                self.std = std[0]
            elif self.noise == 4: # use estimated mixture noise
                filename = '%s/%s' % (conf.exp_dir, self.noise_path)
                result = torch.load(filename)
                self.mean = result['mean'].numpy() / float(conf.res_in - 1)
                self.std = result['std'].numpy() / float(conf.res_in - 1)
                self.weight = result['weight'].numpy()
            else:
                raise ValueError('unsupported noise mode %d' % self.noise)

    def __getitem__(self, index):
        # get 2d/3d pose, bounding box, camera information
        pose2d, bbox, pose3d, cam_f, cam_c = self.get_part_info(index)
        cam_f = cam_f.astype(np.float32)
        cam_c = cam_c.astype(np.float32)

        # SCALING
        pose2d = pose2d * 0.5
        bbox = bbox * 0.5
        cam_f = cam_f * 0.5
        cam_c = cam_c * 0.5

        # data augmentation (flipping)
        if self.split == 'train' and np.random.random() < 0.5:
            pose2d_flip = pose2d.copy()
            for i in range(len(flip_index)):
                pose2d_flip[i] = pose2d[flip_index[i]].copy()
            pose3d_flip = pose3d.copy()
            for i in range(len(flip_index)):
                pose3d_flip[i] = pose3d[flip_index[i]].copy()
            pose2d = pose2d_flip.copy()
            pose3d = pose3d_flip.copy()
            pose2d[:, 0] = self.width - pose2d[:, 0]
            pose3d[:, 0] *= -1

            #bbox[0] = self.width - bbox[0] - bbox[2]
            #cam_c[0] = self.width - cam_c[0]

        # original 2d pose
        meta2d = pose2d.copy()

        # set 2d pose
        if self.noise == 2:
            if not self.split == 'train':
                pose2d = pose2d - bbox[0:2]
                pose2d = pose2d / bbox[2:4]
                pose2d = pose2d * float(conf.res_in - 1)
        else:
            pose2d = pose2d - bbox[0:2]
            pose2d = pose2d / bbox[2:4]
            if self.split == 'train':
                if self.noise == 1:
                    pose2d = pose2d + np.random.randn(pose2d.shape[0], pose2d.shape[1]) * self.std_train
                if self.noise == 3:
                    val = np.random.random((pose2d.shape[0]))
                    pose2d += (np.random.randn(pose2d.shape[0], pose2d.shape[1]) * self.std + self.mean) * (val < self.weight).reshape((pose2d.shape[0], 1))
            elif self.split == 'test':
                if not self.std_test == 0.0:
                    if (self.std_test > 0.0) and (self.std_test < 1.0):
                        # gaussian noise
                        pose2d = pose2d + np.random.randn(pose2d.shape[0], pose2d.shape[1]) * self.std_test
                    elif (self.std_test > 1.0) and (self.std_test < 2.0):
                        # uniform noise
                        val = np.random.random((pose2d.shape[0])) + 1.0
                        pose2d += ((np.random.random((pose2d.shape[0], pose2d.shape[1])) * 100.0 - 50.0) / float(conf.res_in - 1)) * (val < self.opt.std_test).reshape((pose2d.shape[0], 1))
            pose2d = pose2d * float(conf.res_in - 1)

        # root coordinates
        coords_root = pose3d[conf.root].copy()
        depth_root = coords_root[2].copy()
        depth_root_canonical = coords_root[2].copy() / np.sqrt(np.prod(cam_f)) * conf.f0

        # set 3d pose
        pose3d = pose3d - pose3d[conf.root]
        pose3d = np.delete(pose3d, (conf.root), axis=0)

        # set data
        data = {'pose2d': pose2d, 'bbox': bbox,
            'pose3d': pose3d, 'coords_root': coords_root,
            'depth_root': depth_root,
            'depth_root_canonical': depth_root_canonical,
            'cam_f': cam_f, 'cam_c': cam_c,
            'meta2d': meta2d}

        return data

    def __len__(self):
        return self.num_samples

