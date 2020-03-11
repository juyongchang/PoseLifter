from numpy.random import randn
import conf

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def rnd(x):
    return max(-2 * x, min(2 * x, randn() * x))

def flip(img):
    return img[:, :, ::-1].copy()

def shuffle_lr(x):
    for e in conf.flip_index:
        x[e[0]], x[e[1]] = x[e[1]].copy(), x[e[0]].copy()
    return x

def adjust_learning_rate(optimizer, epoch, drop_lr, lr):
    v = lr * (0.1 ** (epoch // drop_lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = v

