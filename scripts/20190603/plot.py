
'''
	Vary the communication time
'''
import matplotlib.pyplot as plt
import numpy as np
import os

from utils import ret_list, ret_time_distribution


name_list = ['STrain', 'SSP s=40', 'BSP', 'ADACOMM', 'Fixed ADACOMM tau=40']
# data_dir = '/Users/hhp/Desktop/Lab_record/new/Lab_record/'
data_dir = os.getenv("DATA_DIR")

# Loss: before adding the sleeping time for the communication function
usp = ret_list(data_dir + '/20190602_01/ps_global_loss_usp.txt')
ssp = ret_list(data_dir + '/20190602_02/ps_global_loss_ssp.txt')
plt.figure(num=4, figsize=(8, 3))
ax = plt.subplot(121)
ax.plot(usp[:, 0], usp[:, -1], label='STrain')
ax.plot(ssp[:, 0], ssp[:, -1], label='SSP s=40')
plt.legend()
plt.ylabel('Global Loss')
plt.xlabel('Time (s)')

ax = plt.subplot(122)
usp = ret_time_distribution(data_dir + '/20190602_01/')
ssp = ret_time_distribution(data_dir + '/20190602_02/')
cmp_list = np.array([usp[0], ssp[0]])
blo_list = np.array([usp[1], ssp[1]])
com_list = np.array([usp[2], ssp[2]])
ax.bar(np.arange(2), cmp_list, label='Computation Time')
ax.bar(np.arange(2), blo_list, label='Blocked Time', bottom=cmp_list)
ax.bar(np.arange(2), com_list, label='Overhead', bottom=cmp_list+blo_list)
plt.ylabel('Nomalized Time')
plt.ylim(0, 1.2)
# plt.xlabel(name_list[i])
plt.xticks(np.arange(2), ['Strain', 'SSP s=40'])
plt.legend()

# plt.title('Time Distribution with STrain')
plt.subplots_adjust(top=0.95, bottom=0.2, left=0.1, right=0.95, hspace=0.4,
                    wspace=0.4)
import os
plt.savefig(os.path.join("fig", os.path.basename(__file__).split(".py")[0] + ".pdf"))


