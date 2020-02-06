from numpy import genfromtxt
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font',size=22)

#plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('jet_r')
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-darkgrid')

my_data = genfromtxt('results/concat-test-l12-10', delimiter=',')
print(my_data.shape)

for i in range(12):
    for j in range(12):
        if j + i > 11:
            my_data[i,j] = -0.05

for i in range(12):
    clen = 12 - i
    xx = np.arange(clen) + i + 2
    val = my_data[i, 0:clen]
    plt.plot(xx, val, color=palette(i/11.0), label='$l_1$=' + str(i+1))

plt.ylim(0, 1.0)
plt.xlabel('$l_2$')
plt.ylabel('validation accuracy')
plt.legend(loc='upper left', fontsize=13, frameon=True)

plt.show()
