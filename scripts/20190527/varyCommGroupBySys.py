'''
	Vary the communication time
'''
import matplotlib.pyplot as plt
import numpy as np

def ret_list(filename):
	lines = [[float(x) for x in line.rstrip('\n').split(',')] for line in open(filename)]
	return np.array(lines)

def ret_accuracy(filename):
	lines = [line.rstrip('\n') for line in open(filename)]
	accuracy = lines[-1].split(',')[-1].split(']')[0]
	return float(accuracy)

def ret_some_time(filename):
	lines = [line.rstrip('\n') for line in open(filename)]
	time = lines[-1].split(':')[-1].split('[')[-1].split(']')[0].split(',')
	return sum([float(_x) for _x in time])/3.0

def normalize(l1, l2, l3, time):
	''' normalize the computation time, commit overhead and blocked time with the total time '''
	a = np.array([l1, l2, l3])
	return a / time

name_list = ['STrain', 'SSP s=40', 'BSP', 'ADACOMM', 'Fixed ADACOMM tau=40']
# data_dir = '/Users/hhp/Desktop/Lab_record/new/Lab_record/'
data_dir = "/Users/hhp/Desktop/STrainData&Record/resultData/Lab_record"

# Loss: before adding the sleeping time for the communication function
usp = ret_list(data_dir + '/20190430_03/ps_global_loss_usp.txt')
bsp = ret_list(data_dir + '/20190418_01/ps_global_loss_ssp.txt')
ssp = ret_list(data_dir + '/20190418_02/ps_global_loss_ssp.txt')
ada = ret_list(data_dir + '/20190422_02/ps_global_loss_ada.txt')
fixed_ada = ret_list(data_dir + '/20190420_03/ps_global_loss_ada.txt')
allList_0 = [usp, ssp, bsp, ada, fixed_ada]
# Loss: after adding the sleeping time 1s for the communication function
usp = ret_list(data_dir + '/20190520_02/ps_global_loss_usp.txt')
bsp = ret_list(data_dir + '/20190522_01/ps_global_loss_ssp.txt')
ssp = ret_list(data_dir + '/20190523_01/ps_global_loss_ssp.txt')
ada = ret_list(data_dir + '/20190524_01/ps_global_loss_ada.txt')
fixed_ada = ret_list(data_dir + '/20190520_03/ps_global_loss_ada.txt')
allList_1 = [usp, ssp, bsp, ada, fixed_ada]
# Loss: after adding the sleeping time 2.5s for the communication function
usp = ret_list(data_dir + '/20190520_01/ps_global_loss_usp.txt')
bsp = ret_list(data_dir + '/20190521_03/ps_global_loss_ssp.txt')
ssp = ret_list(data_dir + '/20190523_02/ps_global_loss_ssp.txt')
ada = ret_list(data_dir + '/20190523_03/ps_global_loss_ada.txt')
fixed_ada = ret_list(data_dir + '/20190519_02/ps_global_loss_ada.txt')
allList_2_5 = [usp, ssp, bsp, ada, fixed_ada]
# Loss: after adding the sleeping time 5s for the communication function
usp = ret_list(data_dir + '/20190518_01/ps_global_loss_usp.txt')
# usp = ret_list(data_dir + '/20190521_02/ps_global_loss_usp.txt')
bsp = ret_list(data_dir + '/20190517_01/ps_global_loss_ssp.txt')
ssp = ret_list(data_dir + '/20190516_02/ps_global_loss_ssp.txt')
ada = ret_list(data_dir + '/20190518_02/ps_global_loss_ada.txt')
fixed_ada = ret_list(data_dir + '/20190519_01/ps_global_loss_ada.txt') #  20190518_03
allList_5 = [usp, ssp, bsp, ada, fixed_ada]
lossList = [allList_0, allList_1, allList_2_5, allList_5]



bsp_nosleep = ret_list(data_dir + '/20190521_01/ps_global_loss_ssp.txt')


# # time - loss
# subtitle = ['No sleep', 'Sleep 2s', 'Sleep 5s', 'Sleep 10s']
# plt.figure(num=4, figsize=(8, 8))
# for i in range(len(name_list)):
# 	ax = plt.subplot(320 + i + 1)
# 	for j in range(4):
# 		ax.plot(lossList[j][i][:, 0], lossList[j][i][:, -1], label=subtitle[j])
# 	plt.ylabel('Global Loss')
# 	plt.xlabel('Time (s) \n' + name_list[i])
# 	plt.legend()
# ax = plt.subplot(326)
# # ax.plot(lossList[0][2][:, 0], lossList[0][2][:, -1], label='Var=0.001')
# ax.plot(bsp_nosleep[:, 0], bsp_nosleep[:, -1], label='Var=0')
# plt.ylabel('Global Loss')
# plt.xlabel('Time (s)\nBSP')
# plt.legend()

# step-loss
subtitle = ['No sleep', 'Sleep 2s', 'Sleep 5s', 'Sleep 10s']
plt.figure(num=4, figsize=(8, 8))
for i in range(len(name_list)):
	ax = plt.subplot(320 + i + 1)
	for j in range(4):
		ax.plot(lossList[j][i][:, 1], lossList[j][i][:, -1], label=subtitle[j])
	plt.ylabel('Global Loss')
	plt.xlabel('Step \n' + name_list[i])
	plt.legend()

# plt.title('Time Distribution with STrain')
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=0.4,
                    wspace=0.4)


import os
plt.savefig(os.path.join("fig", os.path.basename(__file__).split(".py")[0] + ".pdf"))



