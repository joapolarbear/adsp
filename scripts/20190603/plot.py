
'''
	Vary the communication time
'''
import matplotlib.pyplot as plt
import numpy as np
import os

def ret_list(filename):
	lines = [[float(x) for x in line.rstrip('\n').split(',')] for line in open(filename)]
	return np.array(lines)

def ret_accuracy(filename):
	lines = [line.rstrip('\n') for line in open(filename)]
	accuracy = lines[-1].split(',')[-1].split(']')[0]
	return float(accuracy)


def ret_some_time(filename):
	lines = [line.rstrip('\n') for line in open(filename)]
	tmp = lines[-1].split(':')
	time = tmp[-1].split('[')[-1].split(']')[0].split(',')
	return sum([float(_x) for _x in time])/3.0, float(tmp[0])

def normalize(l1, l2, l3, time):
	''' normalize the computation time, commit overhead and blocked time with the total time '''
	a = np.array([l1, l2, l3])
	return a / time

def ret_time_distribution(dirname):
	if(os.path.isfile(dirname + 'ps_cmp_time_usp.txt')):
		cmp_time, total_time = ret_some_time(dirname + 'ps_cmp_time_usp.txt')
		blocked_time, _ = ret_some_time(dirname + 'ps_blocked_time_usp.txt')
	elif(os.path.isfile(dirname + 'ps_cmp_time_ssp.txt')):
		cmp_time, total_time = ret_some_time(dirname + 'ps_cmp_time_ssp.txt')
		blocked_time, _ = ret_some_time(dirname + 'ps_blocked_time_ssp.txt')
	elif(os.path.isfile(dirname + 'ps_cmp_time_ada.txt')):
		cmp_time, total_time = ret_some_time(dirname + 'ps_cmp_time_ada.txt')
		blocked_time, _ = ret_some_time(dirname + 'ps_blocked_time_ada.txt')
	else:
		print('Error: ' + dirname)
		return
	# print(type(total_time), type(cmp_time), type(blocked_time))
	commit_overhead = total_time - cmp_time - blocked_time
	return normalize(cmp_time, blocked_time, commit_overhead, total_time)


name_list = ['STrain', 'SSP s=40', 'BSP', 'ADACOMM', 'Fixed ADACOMM tau=40']
# data_dir = '/Users/hhp/Desktop/Lab_record/new/Lab_record/'
data_dir = "/Users/hhp/Desktop/STrainData&Record/resultData/Lab_record"

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
plt.savefig(os.path.basename(__file__).split(".py")[0] + ".pdf")


