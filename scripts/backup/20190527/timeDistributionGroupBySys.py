'''
	Vary the communication time
'''
import matplotlib.pyplot as plt
import numpy as np
import os

from utils import ret_time_distribution


name_list = ['STrain', 'SSP s=40', 'BSP', 'ADACOMM', 'Fixed ADACOMM tau=40']
# data_dir = '/Users/hhp/Desktop/Lab_record/new/Lab_record/'
data_dir = os.getenv("DATA_DIR")

# Loss: before adding the sleeping time for the communication function
usp = ret_time_distribution(data_dir + '/20190430_03/')
bsp = ret_time_distribution(data_dir + '/20190418_01/')
# ssp = ret_time_distribution(data_dir + '/20190418_02/')
ssp = ret_time_distribution(data_dir + '/20190527_01/')
ada = ret_time_distribution(data_dir + '/20190422_02/')
fixed_ada = ret_time_distribution(data_dir + '/20190420_03/')
allList_0 = [usp, ssp, bsp, ada, fixed_ada]
# Loss: after adding the sleeping time 1s for the communication function
usp = ret_time_distribution(data_dir + '/20190520_02/')
bsp = ret_time_distribution(data_dir + '/20190522_01/')
ssp = ret_time_distribution(data_dir + '/20190523_01/')
ada = ret_time_distribution(data_dir + '/20190524_01/')
fixed_ada = ret_time_distribution(data_dir + '/20190520_03/')
allList_1 = [usp, ssp, bsp, ada, fixed_ada]
# Loss: after adding the sleeping time 2.5s for the communication function
usp = ret_time_distribution(data_dir + '/20190520_01/')
bsp = ret_time_distribution(data_dir + '/20190521_03/')
ssp = ret_time_distribution(data_dir + '/20190523_02/')
ada = ret_time_distribution(data_dir + '/20190523_03/')
fixed_ada = ret_time_distribution(data_dir + '/20190519_02/')
allList_2_5 = [usp, ssp, bsp, ada, fixed_ada]
# Loss: after adding the sleeping time 5s for the communication function
usp = ret_time_distribution(data_dir + '/20190518_01/')
# usp = ret_time_distribution(data_dir + '/20190521_02/')
bsp = ret_time_distribution(data_dir + '/20190517_01/')
ssp = ret_time_distribution(data_dir + '/20190516_02/')
ada = ret_time_distribution(data_dir + '/20190518_02/')
fixed_ada = ret_time_distribution(data_dir + '/20190519_01/') #  20190518_03
allList_5 = [usp, ssp, bsp, ada, fixed_ada]
timeList = [allList_0, allList_1, allList_2_5, allList_5]




subtitle = ['No sleep', 'Sleep 2s', 'Sleep 5s', 'Sleep 10s']
plt.figure(num=4, figsize=(8, 8))
for i in range(len(name_list)):
	ax = plt.subplot(320 + i + 1)
	cmp_list = np.array([timeList[j][i][0] for j in range(4)])
	blo_list = np.array([timeList[j][i][1] for j in range(4)])
	com_list = np.array([timeList[j][i][2] for j in range(4)])
	ax.bar(np.arange(4), cmp_list, label='Computation Time')
	ax.bar(np.arange(4), blo_list, label='Blocked Time', bottom=cmp_list)
	ax.bar(np.arange(4), com_list, label='Overhead', bottom=cmp_list+blo_list)
	plt.ylabel('Nomalized Time')
	plt.ylim(0, 1.2)
	plt.xlabel(name_list[i])
	plt.xticks(np.arange(4), subtitle)
	plt.legend()


# plt.title('Time Distribution with STrain')
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=0.4,
                    wspace=0.4)

import os
plt.savefig(os.path.join("fig", os.path.basename(__file__).split(".py")[0] + ".pdf"))

