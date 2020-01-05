from numpy import genfromtxt
import numpy as np

filenames  = ['e10', 'e20', 'e30', 'e40', 'e50', 'e80', 'e100', 'e180']
x = [64, 64, 128, 
128, 256, 256,
256, 512, 512,
512, 512, 512, 512]

x = [str(item) for item in x] + ['gt']
xx = np.arange(14) + 1

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font',size=22)

plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')
plt.figure(figsize=(18, 6))

cc = 0
for ff in filenames:
    cc += 1
    my_data = genfromtxt('vgg16/vgg16-' + ff, delimiter=',')
    print(my_data.shape)
    print(ff)
    y1 = my_data[:, 1]
    y2 = my_data[:, 2]
    if cc == len(filenames):
        plt.plot(xx, y1, color=palette(0), label='train', alpha=1.0/len(filenames) * (cc))
        plt.plot(xx, y2, color=palette(1), label='test', alpha=1.0/len(filenames) * (cc))
    else:
        plt.plot(xx, y1, color=palette(0), alpha=1.0/len(filenames) * (cc))
        plt.plot(xx, y2, color=palette(1), alpha=1.0/len(filenames) * (cc))
    #plt.xlabel(x)

plt.ylabel('accuracy')
plt.xlabel('grafted layer')
plt.xticks(xx)
plt.ylim(0, 1.0)
plt.legend(loc='lower right', frameon=True)
plt.show()
