import argparse
import os

# Parse cmdline args
parser = argparse.ArgumentParser(description='Image Classification With Two Networks')
parser.add_argument('--logdir', default='logs/vgg16-l2/graft_6.pkl', type=str)
parser.add_argument('--nanase', default=6, type=int)
parser.add_argument('--nlayers', default=16, type=int)
parser.add_argument('--outdir', default='logs/vgg16-l2/', type=str)
args = parser.parse_args()

import pickle
with open(args.logdir, 'rb') as f:
    stat = pickle.load(f)

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font',size=22)

import numpy as np
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')

deg = args.nlayers - args.nanase + 1
for lid in range(args.nanase, args.nlayers + 1):
    lstat_train, lstat_test = stat[str(lid) + 'train'], stat[str(lid) + 'test']

    print(lstat_train['error'].shape)
    nc = lstat_train['error'].shape[0]

    x = (np.arange(nc) + 0.0) / nc
    train_er = lstat_train['error'] / lstat_train['net2_std']
    test_er = lstat_test['error'] / lstat_test['net2_std']
    ind = np.argsort(train_er)
    
    alpha = (args.nlayers - lid + 1.0) / deg
    plt.plot(x, train_er[ind], color=palette(0), label='train', alpha=alpha)
    plt.plot(x, test_er[ind], color=palette(1), label='test', alpha=alpha)
    

plt.savefig(os.path.join(args.outdir, 'graft{}c.pdf'.format(args.nanase)))

