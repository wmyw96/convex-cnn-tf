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

import numpy as np
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set3')

plt.figure(figsize=(10, 8))
import pandas as pd

dfd = {}
deg = args.nlayers - args.nanase + 1

value = []
lyr = []


for lid in range(args.nanase, args.nlayers + 1):
    lstat_train, lstat_test = stat[str(lid) + 'train'], stat[str(lid) + 'test']

    print(lstat_train['error'].shape)
    nc = lstat_train['error'].shape[0]

    x = (np.arange(nc) + 0.0) / nc
    train_er = lstat_train['error'] #/ lstat_train['net2_std']
    test_er = lstat_test['error'] / lstat_test['net2_std']
    ind = np.argsort(train_er)

    alpha = (args.nlayers - lid + 1.0) / deg
    plt.plot(x, train_er[ind], color=palette(lid-args.nanase), label='train' + str(lid))
    #plt.plot(x, test_er[ind], color=palette(lid-args.nanase), label='test' + str(lid), alpha=alpha)
    value += (train_er).tolist()
    lyr += ['layer{}'.format(lid)] * nc

plt.legend(loc='upper right', frameon=True)
plt.savefig(os.path.join(args.outdir, 'graft{}c.pdf'.format(args.nanase)))
plt.close()
plt.clf()

plt.figure(figsize=(10, 8))
import seaborn as sns
df = pd.DataFrame(data={'l2_error': value, 'layer': lyr})
ax = sns.boxplot(x='layer', y='l2_error', data=df, palette="Set3")
ax.set(yscale="log")
plt.savefig(os.path.join(args.outdir, 'graft{}b.pdf'.format(args.nanase)))
plt.close()
plt.clf()