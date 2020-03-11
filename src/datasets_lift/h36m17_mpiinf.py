import torch
import torch.utils.data as data
from h5py import File
import conf
import numpy as np
import pdb

class H36M17_MPIINF(data.Dataset):
    def __init__(self, protocol, split, dense=False, noise=0, std_train=0, std_test=0, noise_path=None):
        print('==> Initializing fusion %s data' % (split))

        annot1 = {}
        tags = ['idx', 'pose2d', 'pose3d', 'bbox', 'cam_f', 'cam_c', 'subject', 'action', 'subaction', 'camera']
        f = File('%s/h36m/h36m17.h5' % (conf.data_dir), 'r')
        for tag in tags:
            annot1[tag] = np.asarray(f[tag]).copy()
        f.close()

        if dense == False:
            idxs = np.mod(annot1['idx'], 50) == 1
            idxs = np.arange(annot1['idx'].shape[0])[idxs]
            for tag in tags:
                annot1[tag] = annot1[tag][idxs]

        idxs = np.full(annot1['idx'].shape[0], False)
        subject = subject_list[protocol][1-int(split=='train' or split=='test_train')]
        for i in range(len(subject)):
            idxs = idxs + (annot1['subject']==subject[i])
        idxs = np.arange(annot1['idx'].shape[0])[idxs]
        for tag in tags:
            annot1[tag] = annot1[tag][idxs]

        annot2 = {}
        tags = ['pose2d', 'pose3d', 'bbox', 'cam_f', 'cam_c', 'subject', 'sequence', 'video']
        f = File('%s/inf/inf_%s.h5' % (conf.data_dir, split), 'r')
        for tag in tags:
            annot2[tag] = np.asarray(f[tag]).copy()
        f.close()

        annot = {}
        annot['pose2d'] = np.concatenate((annot1['pose2d'], annot2['pose2d'].astype(np.float32)), axis=0)
        annot['pose3d'] = np.concatenate((annot1['pose3d'], annot2['pose3d'].astype(np.float32)), axis=0)
        annot['bbox'] = np.concatenate((annot1['bbox'], annot2['bbox']), axis=0)
        annot['cam_f'] = np.concatenate((annot1['cam_f'], annot2['cam_f']), axis=0)
        annot['cam_c'] = np.concatenate((annot1['cam_c'], annot2['cam_c']), axis=0)

        self.split = split
        self.noise = noise
        self.std_train = std_train
        self.std_test = std_test
        self.noise_path = noise_path
        self.annot = annot
        self.num_samples = self.annot['pose2d'].shape[0]

        # load error statistics
        self.load_error_stat()

        print('Load %d fusion %s samples' % (self.num_samples, split))

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
            if self.noise == 2:
                filename = '%s_protocol%d/resnet152-int/fusion/rmsprop_lr1.0e-05_batch48/test_train.pth' % (conf.exp_dir, self.protocol)
                result = torch.load(filename)
                self.annot['pose2d'] = result['pred'].cpu().numpy()
            elif self.noise == 3:
                filename = '../pose2d-hmap-resnet/analysis/stat_fusion_protocol0_simple.pth'
                result = torch.load(filename)
                mean = result['mean'].numpy() / float(conf.res_in - 1)
                self.mean = mean[0]
                std = result['std'].numpy() / float(conf.res_in - 1)
                self.std = std[0]
            elif self.noise == 4:
                filename = '../pose2d-hmap-resnet/analysis/stat_fusion_protocol0_1d_em1.pth'
                result = torch.load(filename)
                self.mean = result['mean'].numpy() / float(conf.res_in - 1)
                self.std = result['std'].numpy() / float(conf.res_in - 1)
                self.weight = result['weight'].numpy()
            elif self.noise == 5:
                filename = '../pose2d-hmap-resnet/analysis/stat_fusion_protocol0_1d_em2.pth'
                result = torch.load(filename)
                self.mean1 = result['mean1'].numpy() / float(conf.res_in - 1)
                self.std1 = result['std1'].numpy() / float(conf.res_in - 1)
                self.mean2 = result['mean2'].numpy() / float(conf.res_in - 1)
                self.std2 = result['std2'].numpy() / float(conf.res_in - 1)
                self.weight = result['weight'].numpy()
            elif self.noise == 6:
                filename = '../pose2d-hmap-resnet/analysis/stat_fusion_protocol0_2d_em1.pth'
                result = torch.load(filename)
                self.mean = result['mean'].numpy() / float(conf.res_in - 1)
                self.std = result['std'].numpy() / float(conf.res_in - 1)
                self.weight = result['weight'].numpy()
            elif self.noise == 7:
                filename = '../pose2d-hmap-resnet/analysis/stat_fusion_protocol0_2d_em2.pth'
                result = torch.load(filename)
                self.mean1 = result['mean1'].numpy() / float(conf.res_in - 1)
                self.std1 = result['std1'].numpy() / float(conf.res_in - 1)
                self.mean2 = result['mean2'].numpy() / float(conf.res_in - 1)
                self.std2 = result['std2'].numpy() / float(conf.res_in - 1)
                self.weight = result['weight'].numpy()
            elif self.noise == 8:
                filename = '%s/%s' % (conf.exp_dir, self.noise_path)
                result = torch.load(filename)
                self.mean = result['mean'].numpy() / float(conf.res_in - 1)
                self.std = result['std'].numpy() / float(conf.res_in - 1)
                self.weight = result['weight'].numpy()

    def __getitem__(self, index):
        # get 2D/3D pose, bounding box, camera information
        pose2d, bbox, pose3d, cam_f, cam_c = self.get_part_info(index)
        cam_f = cam_f.astype(np.float32)
        cam_c = cam_c.astype(np.float32)

        # original 2D pose
        meta2d = pose2d.copy()

        # set 2D pose
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
                    pose2d = pose2d + np.random.randn(pose2d.shape[0], pose2d.shape[1]) * self.std + self.mean
                elif self.noise == 4:
                    val = np.random.random((pose2d.shape[0], pose2d.shape[1]))
                    pose2d += (np.random.randn(pose2d.shape[0], pose2d.shape[1]) * self.std + self.mean) * (val < self.weight)
                    pose2d += ((np.random.random((pose2d.shape[0], pose2d.shape[1])) * 200.0 - 100.0) / float(conf.res_in - 1)) * (val >= self.weight)
                elif self.noise == 5:
                    val = np.random.random((pose2d.shape[0], pose2d.shape[1]))
                    pose2d += (np.random.randn(pose2d.shape[0], pose2d.shape[1]) * self.std1 + self.mean1) * (val < self.weight)
                    pose2d += (np.random.randn(pose2d.shape[0], pose2d.shape[1]) * self.std2 + self.mean2) * (val >= self.weight)
                elif self.noise == 6:
                    val = np.random.random((pose2d.shape[0]))
                    pose2d += (np.random.randn(pose2d.shape[0], pose2d.shape[1]) * self.std + self.mean) * (val < self.weight).reshape((pose2d.shape[0], 1))
                    pose2d += ((np.random.random((pose2d.shape[0], pose2d.shape[1])) * 100.0 - 50.0) / float(conf.res_in - 1)) * (val >= self.weight).reshape((pose2d.shape[0], 1))
                elif self.noise == 7:
                    val = np.random.random((pose2d.shape[0]))
                    pose2d += (np.random.randn(pose2d.shape[0], pose2d.shape[1]) * self.std1 + self.mean1) * (val < self.weight).reshape((pose2d.shape[0], 1))
                    pose2d += (np.random.randn(pose2d.shape[0], pose2d.shape[1]) * self.std2 + self.mean2) * (val >= self.weight).reshape((pose2d.shape[0], 1))
                elif self.noise == 8:
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

