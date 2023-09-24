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

name_list = ['ADSP', 'SSP s=40', 'BSP', 'ADACOMM', '\nFixed ADACOMM\n tau=40']
# data_dir = 'C:/Users/Admin/Dropbox/Lab_record/new/Lab_record/'
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

# subtitle = ['No sleep', 'Sleep 2s', 'Sleep 5s', 'Sleep 10s']
# plt.figure(num=4, figsize=(8, 8))
# for i in range(4):
# 	ax = plt.subplot(220 + i + 1)
# 	for j in range(len(lossList[i])):
# 		ax.plot(lossList[i][j][:, 0], lossList[i][j][:, -1], label=name_list[j])
# 	plt.ylabel('Global Loss')
# 	plt.xlabel('Time (s) \n' + subtitle[i])
# 	plt.legend()

subtitle = ['(a) Delay = 0s', '(b) Delay = 2s', '(c) Delay = 5s', '(d) Delay = 10s']
plt.figure(num=4, figsize=(8, 6))
for i in range(4):
	ax = plt.subplot(220 + i + 1)
	for j in range(len(lossList[i])):
		ax.plot(lossList[i][j][:, 0], lossList[i][j][:, -1], label=name_list[j])
	plt.ylabel('Global Loss')
	plt.xlabel('Wall-Clock Time (s) \n' + subtitle[i])
	if(i == 0):
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
		plt.legend(ncol=5, bbox_to_anchor=(0, 1.23), loc='center left')


# plt.title('Time Distribution with STrain')
plt.subplots_adjust(top=0.8, bottom=0.1, left=0.1, right=0.95, hspace=0.4,
                    wspace=0.3)
plt.savefig("fig/vary_network_delay.pdf")


