import argparse
import os

# Parse cmdline args
parser = argparse.ArgumentParser(description='Image Classification With Two Networks')
parser.add_argument('--logdir', default='logs/vgg16-l2-51/', type=str)
parser.add_argument('--nl', default=12, type=int)

args = parser.parse_args()

import pickle

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import numpy as np
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set3')

plt.figure(figsize=(10, 8))
import pandas as pd

val_tr = []
val_te = []
lyr = []

for lid in range(2, args.nl + 1):
    with open(os.path.join(args.logdir, 'graft_{}.pkl'.format(lid)), 'rb') as f:
        stat = pickle.load(f)
    lstat_train, lstat_test = stat[str(lid) + 'train'], stat[str(lid) + 'test']
    lyr.append(lid)
    val_tr.append(lstat_train['graft_acc'])
    val_te.append(lstat_test['graft_acc'])

print(val_tr)
print(val_te)
lyr = np.array(lyr)
val_tr = np.array(val_tr)
val_te = np.array(val_te)
plt.plot(lyr, val_tr, color=palette(0), label='train')
plt.plot(lyr, val_te, color=palette(1), label='test')
plt.ylim(0, 1)
plt.xlabel('grafted layer')
plt.ylabel('accuracy')
plt.legend(loc='upper right', frameon=True)
plt.show()
plt.clf()

plt.figure(figsize=(10, 8))
for lid in range(2, args.nl + 1):
    with open(os.path.join(args.logdir, 'graft_{}.pkl'.format(lid)), 'rb') as f:
        stat = pickle.load(f)

    yv = []
    for llid in range(lid, 17):
        lstat_train, lstat_test = stat[str(llid) + 'train'], stat[str(llid) + 'test']
        yv.append(np.mean(lstat_train['error'] / lstat_train['net2_std']))

    plt.plot(np.arange(17 - lid) + lid, yv, color=palette(lid - 2), label='graft{}'.format(lid))
plt.legend(loc='upper left', frameon=True)
plt.show()
