'''
	Vary the communication time
'''
import matplotlib.pyplot as plt
import numpy as np
import os

from utils import *


def normalize(l1, l2, l3, time):
	''' normalize the computation time, commit overhead and blocked time with the total time '''
	a = np.array([l1, l2, l3])
	return a / time


# data_dir = 'C:/Users/Admin/Dropbox/Lab_record/new/Lab_record/'
data_dir = os.getenv("DATA_DIR")

# Loss: after adding the sleeping time 5s for the communication function
usp = ret_list(data_dir + '/20190603_02/ps_global_loss_usp.txt')
# usp = ret_list(data_dir + '/20190521_02/ps_global_loss_usp.txt')
bsp = ret_list(data_dir + '/20190604_01/ps_global_loss_ssp.txt')
ssp = ret_list(data_dir + '/20190604_02/ps_global_loss_ssp.txt')
ada = ret_list(data_dir + '/20190608_01/ps_global_loss_ada.txt')
# fixed_ada = ret_list(data_dir + '/20190607_01/ps_global_loss_ada.txt') 
fixed_ada = ret_list(data_dir + '/20190609_01/ps_global_loss_ada.txt') 
allList = [usp, ssp, bsp, 
	ada, 
	fixed_ada
	]
name_list = ['ADSP', 'SSP s=40', 'BSP', 
	'ADACOMM', 
	'Fixed ADACOMM\n tau=40'
	]

plt.figure(num=4, figsize=(8, 4))
ax = plt.subplot(111)
for j in range(len(allList)):
	ax.plot(allList[j][:, 0], allList[j][:, -1], label=name_list[j], 
        linestyle=linestyle_str[j][1], linewidth=linewidth)
plt.ylabel('Global Loss')
plt.xlabel('Wall-Clock Time (s) \n')
plt.ylim(10.9, 11.2)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
# plt.legend(ncol=5, bbox_to_anchor=(0, 1.23), loc='center left')
plt.legend()

# plt.title('Time Distribution with STrain')
# plt.subplots_adjust(top=0.93, bottom=0.16, left=0.1, right=0.95, hspace=0.4,
#                     wspace=0.3)
plt.tight_layout()
plt.savefig("fig/large_model_vgg.pdf")


