r = ['1', '4', '16', '24', '28']
acc = [0.9229, 0.9174, 0.88, 0.8672, 0.8552]
prefix = 'logs/c10-vgg16-l2-'

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import os
import pickle
import numpy as np
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set3')
plt.rc('font',size=22)


plt.figure(figsize=(10, 6))
import pandas as pd

def tod(x):
    if x[0] == 'z':
        return '0.'+x[1:]
    else:
        return x

for (i, num_str) in enumerate(r):
    logdir = prefix + num_str
    print(i)
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
    y = acc[i] - np.array(val_te) 
    lyr = np.array(lyr)
    plt.plot(lyr, y, color=palette(i), label='w={}$w_0$'.format(tod(num_str)))

plt.ylim(-0.1, 0.6)
plt.xlabel('grafted layer')
plt.ylabel('increased validation error')
plt.legend(loc='upper left', frameon=True)
plt.show()
plt.clf()