from numpy import genfromtxt
import numpy as np

my_data = genfromtxt('vgg16-l12', delimiter=',')
x = [64, 64, 128, 
128, 256, 256,
256, 512, 512,
512, 512, 512, 512]

x = [str(item) for item in x]
xx = np.arange(13) + 1
y1 = my_data[:, 1]
y2 = my_data[:, 2]

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font',size=22)

plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')
plt.figure(figsize=(10, 6))

plt.plot(xx, y1, color=palette(0), label='train')
plt.plot(xx, y2, color=palette(1), label='test')
#plt.xlabel(x)
plt.ylabel('accuracy')
plt.xlabel('grafted layer')
plt.xticks(xx)
plt.ylim(0, 1.0)
plt.legend(loc='lower right', frameon=True)
plt.show()
