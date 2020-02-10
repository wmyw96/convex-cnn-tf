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
plt.rc('font',size=20)
#plt.gcf().subplots_adjust(bottom=0.0)
plt.rc('figure', autolayout=True)

#plt.tight_layout()

plt.figure(figsize=(10, 4))
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
    plt.plot(lyr - 1, y * 100, color=palette(i), label='$\lambda$={}$w_0$'.format(tod(num_str)), marker='o')

plt.ylim(-10, 60)
plt.xlim(0, 14.5)
plt.xticks([2, 4, 6, 8, 10])
plt.xlabel('grafted layer')
plt.ylabel('increased valid error (%)')
plt.legend(loc='upper right', fontsize=18, frameon=True)

plt.show()
plt.clf()