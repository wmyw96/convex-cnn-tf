from numpy import genfromtxt
import numpy as np

xx = np.arange(13) + 1

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font',size=22)

#plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')
plt.figure(figsize=(8, 8))

my_data = genfromtxt('concat2-train', delimiter=',')
print(my_data.shape)

for i in range(13):
    for j in range(13):
        if j + i > 12:
            my_data[i,j] = -0.05

plt.matshow(my_data)
plt.ylabel('i')
plt.xlabel('c')
#plt.xticks(xx)
#plt.legend(loc='lower right', frameon=True)
plt.show()
