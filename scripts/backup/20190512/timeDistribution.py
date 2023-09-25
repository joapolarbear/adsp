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


# 20190430_03: Cifar + STrain
commit_overhead = [74.635, 70.738, 76.517]
blocked_time = [96.650, 97.919, 62.379]
computation_time = [2023.684, 2023.695, 2059.416]

commit_overhead = (2225.8325 - np.array(computation_time) - np.array(blocked_time))
Cifar10 = normalize(computation_time, commit_overhead, blocked_time, 2225.8325)

# 20190512_01: ImageNet + STrain
commit_overhead = [914.45, 877.615, 729.481]
blocked_time = [995.619, 946.722, 834.719]
computation_time = [27828.566, 27850.477, 28200.551]

commit_overhead = (30421.037 - np.array(computation_time) - np.array(blocked_time))
ImageNet = normalize(computation_time, commit_overhead, blocked_time, 30421.037)


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
plt.xticks(np.arange(3)+bar_width/2, ['WK0', 'WK1', 'WK2'])
# plt.xlabel('Time (s)')

plt.title('Time Distribution with STrain')
plt.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.95, hspace=0.5,
                    wspace=0.3)

import os
plt.savefig(os.path.join("fig", os.path.basename(__file__).split(".py")[0] + ".pdf"))


