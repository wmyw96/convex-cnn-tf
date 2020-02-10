r = ['4', '8', '10']
#acc = [0.9175, 0.9036, 0.8883, 0.8522]
acc = [0.9036, 0.8883, 0.8522]

prefix = 'logs/c10-vgg16-l12-'

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import os
import pickle
import numpy as np
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')
plt.rc('font',size=20)
plt.rc('figure', autolayout=True)


plt.figure(figsize=(10, 4))
import pandas as pd


def concat(x, y):
    if len(x) == 1 and len(y) == 1:
        return x + y
    else:
        return x + '-' + y


for (i, num_str) in enumerate(r):
    logdir = prefix + concat('1', num_str)
    val_tr = []
    val_te = []
    lyr = []
    for lid in range(2, 13):
        with open(os.path.join(logdir, 'graft_{}.pkl'.format(lid)), 'rb') as f:
            stat = pickle.load(f)
        lstat_train, lstat_test = stat[str(lid) + 'train'], stat[str(lid) + 'test']
        lyr.append(lid)
        val_tr.append(lstat_train['graft_acc'])
        val_te.append(lstat_test['graft_acc'])
    print(val_tr)
    print(val_te)
    y = (acc[i] - np.array(val_te)) * 100 
    lyr = np.array(lyr)
    plt.plot(lyr - 1, y, color=palette(i), label='{}'.format('1-{}'.format(num_str)), 
        linestyle='-', marker='o')

    logdir = prefix + concat(num_str, '1')
    val_tr = []
    val_te = []
    lyr = []
    for lid in range(2, 13):
        with open(os.path.join(logdir, 'graft_{}.pkl'.format(lid)), 'rb') as f:
            stat = pickle.load(f)
        lstat_train, lstat_test = stat[str(lid) + 'train'], stat[str(lid) + 'test']
        lyr.append(lid)
        val_tr.append(lstat_train['graft_acc'])
        val_te.append(lstat_test['graft_acc'])
    print(val_tr)
    print(val_te)
    y = (0.9192 - np.array(val_te)) * 100 
    lyr = np.array(lyr)
    plt.plot(lyr - 1, y, color=palette(i), label='{}'.format('{}-1'.format(num_str)), 
        linestyle=':', marker='s')



plt.ylim(-10, 80)
plt.xlim(0, 14.5)
plt.xlabel('grafted layer')
plt.xticks([2, 4, 6, 8, 10])
plt.ylabel('increased valid error (%)')
plt.legend(loc='upper right', fontsize=18, frameon=True)
plt.show()
plt.clf()