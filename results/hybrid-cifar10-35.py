from numpy import genfromtxt
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font',size=22)

#plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')
plt.figure(figsize=(6, 10))
plt.style.use('seaborn-darkgrid')

my_data = genfromtxt('results/linear_inter', delimiter=',')
print(my_data.shape)

xx = my_data[:, 0]
overall = my_data[:, 1]
ce_loss = my_data[:, 2]
reg_loss = my_data[:, 3]

plt.plot(xx, overall, color=palette(0), label='overall')
plt.plot(xx, ce_loss, color=palette(1), label='cross entropy')
plt.plot(xx, reg_loss, color=palette(2), label='regularizer')

plt.xlim(0.0, 1.0)
#plt.ylim(0.0, 2.5)
plt.ylim(0.0, 5.0)
plt.xlabel('$\gamma$')
plt.ylabel('loss')
#plt.legend(loc='lower right', fontsize=20, frameon=True)

plt.show()
