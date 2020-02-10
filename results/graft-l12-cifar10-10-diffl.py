from numpy import genfromtxt
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font',size=20)

#plt.style.use('seaborn-darkgrid')
plt.rc('figure', autolayout=True)
palette = plt.get_cmap('jet_r')
plt.figure(figsize=(10, 5.5))
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
    plt.plot(xx-1, (1-val)*100, color=palette(i/11.0), marker='o', label='$l_1$=' + str(i+1))


plt.plot(np.arange(0, 16), np.array([14.8] * 16), linestyle=':', linewidth=0.8, color='black')
plt.ylim(0, 100)
plt.xlim(0, 15)
plt.xlabel('$l_2$')
plt.yticks([0.0, 14.8, 20, 40, 60, 80, 100])
plt.xticks([0, 2, 4, 6, 8, 10, 12])
plt.ylabel('valid error (%)')
plt.legend(loc='upper right', fontsize=13, frameon=True)

plt.show()
