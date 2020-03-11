
# annot directory
data_dir = './annot'

# image directory
h36m_img_dir = './data/h36m'
mpii_img_dir = './data/mpii'
inf_img_dir = './data/inf'

# experiment directory
exp_dir = './exp'

# number of threads
num_threads = 4

# number of joints
num_joints = 17

# root index (hip)
root = 0

# input/output resolutions
res_in = 256
res_out = 64

# standard deviation of gaussian function for heatmap generation
std = 1.0

# parameters for data augmentation
scale = 0.25
rotate = 30
flip_index = [[3, 6], [2, 5], [1, 4],
              [16, 13], [15, 12], [14, 11]]

# joint index mapping from MPII to H36M
inds = [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13]

# sum of bone lengths (mm)
sum_of_bone_length = 4139.15

# bone structure
bone = [[0, 1], [1, 2], [2, 3],
        [0, 4], [4, 5], [5, 6],
        [0, 7], [7, 8], [8, 9], [9, 10],
        [8, 11], [11, 12], [12, 13],
        [8, 14], [14, 15], [15, 16]]

# number of actions for human3.6m dataset
num_actions = 15

# original image size
width = 1000
height = 1000

# for canonical depth
f0 = 1000

