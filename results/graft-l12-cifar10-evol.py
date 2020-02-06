#r = ['z5', '1', '2', '4', '8', '10']
#acc = [0.9168, 0.9192, 0.9175, 0.9036, 0.8883, 0.8522]
prefix = 'logs/tl-c10-vgg16-l12-e'
acc = [0.314, 0.569, 0.741, 0.789, 0.842, 0.870, 0.877, 0.919]

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import os
import pickle
import numpy as np
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('jet_r')
plt.rc('font',size=22)


plt.figure(figsize=(10, 6))
import pandas as pd

def tod(x):
    if x[0] == 'z':
        return '0.'+x[1:]
    else:
        return x

for (i, num_epoch) in enumerate([1, 2, 5, 8, 20, 40, 60]):
    logdir = prefix + str(num_epoch) + 'f'
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

    y = np.array(val_te) 
    lyr = np.array(lyr)
    plt.plot(lyr, y, color=palette(i/6.0), label='e={}'.format(num_epoch))
    plt.plot(np.arange(1, 14), np.array([acc[i]] * 13), color=palette(i/6.0), linestyle=':', linewidth=1.0)


plt.plot(np.arange(1, 14), np.array([0.9192] * 13), color='black', linestyle=':', linewidth=1.0)

plt.ylim(0.0, 1.0)
plt.xlim(1, 13)
plt.yticks(acc)
plt.xlabel('grafted layer')
plt.ylabel('accuracy')
plt.legend(loc='lower left', frameon=True)
plt.show()
plt.clf()