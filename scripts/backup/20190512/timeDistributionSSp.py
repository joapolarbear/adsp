'''
	plot the difference of the time distribution 
	including computation time, commit overhead and blocked time, 
	between Cifar10 small model (1 MB) and ImageNet VGG model (528 MB)
'''
import matplotlib.pyplot as plt
import numpy as np


def normalize(l1, l2, l3, time):
	a = np.array([l1, l2, l3])
	return a / time

# 20190514_02: Cifar + SSP
commit_overhead = [2596.3977756500244, 2673.3554213047028, 1078.9794418811798]
blocked_time = [716.9461526870728, 2675.398068666458, 0.0002429485321044922]
computation_time = [526.4196655750275, 439.19713377952576, 2071.741060733795]

commit_overhead = (3175.9008259773 - np.array(computation_time) - np.array(blocked_time))
Cifar10 = normalize(computation_time, commit_overhead, blocked_time, 3175.9008259773)

# 20190514_01: ImageNet + SSP
commit_overhead = [15459.625036716461, 17523.742409467697, 15818.832042455673]
blocked_time = [3.170967102050781e-05, 0.00026726722717285156, 8.702278137207031e-05]
computation_time = [20421.181673526764, 18669.172875881195, 20840.232418060303]

commit_overhead = (38191.7214510441 - np.array(computation_time) - np.array(blocked_time))
ImageNet = normalize(computation_time, commit_overhead, blocked_time, 38191.7214510441)


plt.figure(num=4, figsize=(8, 6))
bar_width = 0.2

ax = plt.subplot(111)
ax.bar(range(3), Cifar10[0, :], label='Cifar: Computation time', width=bar_width)
ax.bar(range(3), Cifar10[1, :], label='Cifar: Overhead', width=bar_width, bottom=Cifar10[0, :])
ax.bar(range(3), Cifar10[2, :], label='Cifar: Blocked time', width=bar_width, bottom=Cifar10[0, :] + Cifar10[1, :])

ax.bar(np.array(range(3)) + 1.5 * bar_width, ImageNet[0, :], label='ImageNet: Computation time', width=bar_width)
ax.bar(np.array(range(3)) + 1.5 * bar_width, ImageNet[1, :], label='ImageNet: Overhead', width=bar_width, bottom=ImageNet[0, :])
ax.bar(np.array(range(3)) + 1.5 * bar_width, ImageNet[2, :], label='ImageNet: Blocked time', width=bar_width, bottom=ImageNet[0, :] + ImageNet[1, :])

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
plt.legend(ncol=2, bbox_to_anchor=(0, -0.15), loc='center left')
plt.ylabel('Normalized Time')
plt.ylim(0, 1.25)
plt.xticks(np.arange(3)+bar_width/2, ['WK0', 'WK1', 'WK2'])
# plt.xlabel('Time (s)')

plt.title('Time Distribution with SSP')
plt.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.95, hspace=0.5,
                    wspace=0.3)

import os
plt.savefig(os.path.join("fig", os.path.basename(__file__).split(".py")[0] + ".pdf"))


