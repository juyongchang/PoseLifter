import torch.utils.data as data
from datasets_2d.mpiinf import MPIINF
from datasets_2d.mpii import MPII

class MPIINF_MPII(data.Dataset):
    def __init__(self, split):
        self.split = split
        self.INF = MPIINF(split)
        self.MPII = MPII()
        self.num_inf = len(self.INF)
        self.num_mpii = len(self.MPII)
        print('Load %d MPIINF and %d MPII samples' % (self.num_inf, self.num_mpii))

    def __getitem__(self, index):
        if index < self.num_mpii:
            return self.INF[index]
        else:
            return self.MPII[index - self.num_mpii]

    def __len__(self):
        return self.num_mpii*2

