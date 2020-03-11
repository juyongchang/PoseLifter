
# install cuda 10.2
# install anaconda3

# create conda environment
conda create --name poselifter --clone base

# activate new environment
conda activate poselifter

# install pytorch 1.1.0 for cuda 10.2
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

# install packages
pip install -r requirements.txt



