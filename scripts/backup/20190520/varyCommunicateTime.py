'''
	Vary the communication time
'''
import matplotlib.pyplot as plt
import numpy as np
import os

from utils import ret_list, ret_accuracy, ret_some_time, normalize

data_dir = os.getenv("DATA_DIR")

name_list = ['STrain', 'SSP s=40', 'BSP', 'ADACOMM', '\nFixed ADACOMM\n tau=40']

# Accuracy: before adding the sleeping time for the communication function
usp = ret_accuracy(data_dir + '/20190430_03/ps_global_eval_usp.txt')
bsp = ret_accuracy(data_dir + '/20190418_01/ps_global_eval_ssp.txt')
ssp = ret_accuracy(data_dir + '/20190418_02/ps_global_eval_ssp.txt')
ada = ret_accuracy(data_dir + '/20190422_02/ps_global_eval_ada.txt')
fixed_ada = ret_accuracy(data_dir + '/20190420_03/ps_global_eval_ada.txt')
accuracyList_0 = [usp, ssp, bsp, ada, fixed_ada]
# Accuracy: after adding the sleeping time for the communication function
usp = ret_accuracy(data_dir + '/20190518_01/ps_global_eval_usp.txt')
bsp = ret_accuracy(data_dir + '/20190517_01/ps_global_eval_ssp.txt')
ssp = ret_accuracy(data_dir + '/20190516_02/ps_global_eval_ssp.txt')
ada = ret_accuracy(data_dir + '/20190518_02/ps_global_eval_ada.txt')
fixed_ada = ret_accuracy(data_dir + '/20190519_01/ps_global_eval_ada.txt') # 20190519_01
accuracyList_5 = [usp, ssp, bsp, ada, fixed_ada]

# Loss: before adding the sleeping time for the communication function
usp = ret_list(data_dir + '/20190430_03/ps_global_loss_usp.txt')
bsp = ret_list(data_dir + '/20190418_01/ps_global_loss_ssp.txt')
ssp = ret_list(data_dir + '/20190418_02/ps_global_loss_ssp.txt')
ada = ret_list(data_dir + '/20190422_02/ps_global_loss_ada.txt')
fixed_ada = ret_list(data_dir + '/20190420_03/ps_global_loss_ada.txt')
allList_0 = [usp, ssp, bsp, ada, fixed_ada]
# Loss: after adding the sleeping time for the communication function
usp = ret_list(data_dir + '/20190518_01/ps_global_loss_usp.txt')
# usp = ret_list(data_dir + '/20190521_02/ps_global_loss_usp.txt')
bsp = ret_list(data_dir + '/20190517_01/ps_global_loss_ssp.txt')
ssp = ret_list(data_dir + '/20190516_02/ps_global_loss_ssp.txt')
ada = ret_list(data_dir + '/20190518_02/ps_global_loss_ada.txt')
fixed_ada = ret_list(data_dir + '/20190519_01/ps_global_loss_ada.txt') #  20190518_03
allList_5 = [usp, ssp, bsp, ada, fixed_ada]


usp_2_5 = ret_list(data_dir + '/20190520_01/ps_global_loss_usp.txt')
usp_1 = ret_list(data_dir + '/20190520_02/ps_global_loss_usp.txt')
uspList = [allList_0[0], usp_1, usp_2_5, allList_5[0]]
fixed_ada_2_5 = ret_list(data_dir + '/20190519_02/ps_global_loss_ada.txt') 
# fixed_ada_1 = ret_list(data_dir + '/20190520_03/ps_global_loss_ada.txt') 
fixed_adaList = [allList_0[-1], 
		# fixed_ada_1,
 		fixed_ada_2_5, allList_5[-1]]


usp = ret_some_time(data_dir + '/20190430_03/ps_cmp_time_usp.txt')
bsp = ret_some_time(data_dir + '/20190418_01/ps_cmp_time_ssp.txt')
ssp = ret_some_time(data_dir + '/20190418_02/ps_cmp_time_ssp.txt')
ada = ret_some_time(data_dir + '/20190422_02/ps_cmp_time_ada.txt') # 20190519_01
fixed_ada = ret_some_time(data_dir + '/20190420_03/ps_cmp_time_ada.txt')
cmp_time_0 = np.array([usp, ssp, bsp, ada, fixed_ada])
usp = ret_some_time(data_dir + '/20190518_01/ps_cmp_time_usp.txt')
bsp = ret_some_time(data_dir + '/20190517_01/ps_cmp_time_ssp.txt')
ssp = ret_some_time(data_dir + '/20190516_02/ps_cmp_time_ssp.txt')
ada = ret_some_time(data_dir + '/20190518_02/ps_cmp_time_ada.txt')
fixed_ada = ret_some_time(data_dir + '/20190519_01/ps_cmp_time_ada.txt') #  20190518_03
cmp_time_5 = np.array([usp, ssp, bsp, ada, fixed_ada])
usp = ret_some_time(data_dir + '/20190430_03/ps_blocked_time_usp.txt')
bsp = ret_some_time(data_dir + '/20190418_01/ps_blocked_time_ssp.txt')
ssp = ret_some_time(data_dir + '/20190418_02/ps_blocked_time_ssp.txt')
ada = ret_some_time(data_dir + '/20190422_02/ps_blocked_time_ada.txt')
fixed_ada = ret_some_time(data_dir + '/20190420_03/ps_blocked_time_ada.txt') # 20190519_01
blocked_time_0 = np.array([usp, ssp, bsp, ada, fixed_ada])
usp = ret_some_time(data_dir + '/20190518_01/ps_blocked_time_usp.txt')
bsp = ret_some_time(data_dir + '/20190517_01/ps_blocked_time_ssp.txt')
ssp = ret_some_time(data_dir + '/20190516_02/ps_blocked_time_ssp.txt')
ada = ret_some_time(data_dir + '/20190518_02/ps_blocked_time_ada.txt')
fixed_ada = ret_some_time(data_dir + '/20190519_01/ps_blocked_time_ada.txt') #  20190518_03
blocked_time_5 = np.array([usp, ssp, bsp, ada, fixed_ada])

converge_time_0 = np.array([l[-1, 0] for l in allList_0])
converge_time_5 = np.array([l[-1, 0] for l in allList_5])

commit_overhead_0 = converge_time_0 - cmp_time_0 - blocked_time_0
commit_overhead_5 = converge_time_5 - cmp_time_5 - blocked_time_5


# Cifar10 = normalize(computation_time, commit_overhead, blocked_time, 2225.8325)

plt.figure(num=4, figsize=(8, 8))
ax = plt.subplot(321)
for i in range(len(allList_0)):
	ax.plot(allList_0[i][:, 0], allList_0[i][:, -1], label=name_list[i])
plt.ylabel('Global Loss')
plt.xlabel('Time (s)\nNo sleep')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
plt.legend(ncol=5, bbox_to_anchor=(0, 1.15), loc='center left')


ax = plt.subplot(322)
for i in range(len(allList_5)):
	ax.plot(allList_5[i][:, 0], allList_5[i][:, -1], label=name_list[i])
plt.ylabel('Global Loss')
plt.xlabel('Time (s)\nSleep 10s')



# loss - step
ax = plt.subplot(323)
for i in range(len(allList_0)):
	ax.plot(allList_0[i][:, 1], allList_0[i][:, -1], label=name_list[i])
plt.ylabel('Global Loss')
plt.xlabel('# of step\nNo sleep')
plt.xlim(0, 8000)

# loss - step
ax = plt.subplot(324)
for i in range(len(allList_5)):
	ax.plot(allList_5[i][:, 1], allList_5[i][:, -1], label=name_list[i])
plt.ylabel('Global Loss')
plt.xlabel('# of step\nSleep 10s')
plt.xlim(0, 8000)



'''
# convergence time 
ax = plt.subplot(323)
bar_width = 0.2
converge_time_0_norm = np.array([l[-1, 0] / allList_0[0][-1, 0] for l in allList_0])
converge_time_5_norm = np.array([l[-1, 0] / allList_5[0][-1, 0] for l in allList_5])
ax.bar(np.arange(len(cmp_time_5)) - bar_width/2, converge_time_0_norm, label='No sleep', width=bar_width)
ax.bar(np.arange(len(cmp_time_5)) + bar_width/2, converge_time_5_norm, label='Sleep 10s', width=bar_width)
# ax.bar(range(len(cmp_time_5)), commit_overhead_5, label='Overhead', width=bar_width, bottom=cmp_time_5)
# ax.bar(range(len(cmp_time_5)), blocked_time_5, label='Blocked time', width=bar_width, bottom=cmp_time_5 + commit_overhead_5)
plt.legend()
plt.ylabel('Normalized convergence time')
plt.xticks(np.arange(len(cmp_time_5))+bar_width/2, name_list)

# accuracy
ax = plt.subplot(324)
final_loss_0_norm = np.array([l[-1, 0] / allList_0[0][-1, 0] for l in allList_0])
final_loss_5_norm = np.array([l[-1, 0] / allList_5[0][-1, 0] for l in allList_5])
ax.bar(np.arange(len(cmp_time_5)) - bar_width/2, accuracyList_0, label='No sleep', width=bar_width)
ax.bar(np.arange(len(cmp_time_5)) + bar_width/2, accuracyList_5, label='Sleep 10s', width=bar_width)
# ax.bar(range(len(cmp_time_5)), commit_overhead_5, label='Overhead', width=bar_width, bottom=cmp_time_5)
# ax.bar(range(len(cmp_time_5)), blocked_time_5, label='Blocked time', width=bar_width, bottom=cmp_time_5 + commit_overhead_5)
# plt.legend()
plt.ylabel('Accuracy')
plt.xticks(np.arange(len(cmp_time_5))+bar_width/2, name_list)
'''


'''
# loss - sleep time with STrain
ax = plt.subplot(337)
_name_list = ['sleep 0s', 'sleep 2s', 'sleep 5s', 'sleep 10s']
for i in range(len(uspList)):
	ax.plot(uspList[i][:, 0], uspList[i][:, -1], label=_name_list[i])
plt.ylabel('Global Loss')
plt.xlabel('Time (s)')
plt.xlim(0, 10000)
plt.legend(title='For STrain\'s Communication', fontsize=8)

# loss - sleep time with Fixed
ax = plt.subplot(338)
_name_list = ['sleep 0s', 
# 'sleep 2s', 
'sleep 5s', 'sleep 10s']
for i in range(len(fixed_adaList)):
	ax.plot(fixed_adaList[i][:, 0], fixed_adaList[i][:, -1], label=_name_list[i])
plt.ylabel('Global Loss')
plt.xlabel('Time (s)')
plt.xlim(0, 10000)
plt.legend(title='For Fixed\'s Communication', fontsize=8)

# convergence time - sleep time
ax = plt.subplot(339)
convegence_time = [l[-1, 0] for l in uspList]
convegence_time_2 = [l[-1, 0] for l in fixed_adaList]
final_loss = [l[-1, -1] for l in uspList]
ax.plot([0, 2, 5, 10], convegence_time, label='STrain')
ax.plot([0, 5, 10], convegence_time_2, label='Fixed Ada')
plt.xlabel('Sleep Time per\ncommunication (s)')
plt.legend(fontsize=8)
# plt.ylabel('')
'''


# two test for the same configuration: STrain + sleep 10s communication
usp = ret_list(data_dir + '/20190518_01/ps_global_loss_usp.txt')
usp2 = ret_list(data_dir + '/20190521_02/ps_global_loss_usp.txt')

ax = plt.subplot(325)
ax.plot(usp[:, 0], usp[:, -1])
ax.plot(usp2[:, 0], usp2[:, -1])

ax = plt.subplot(326)
ax.plot(usp[:, 1], usp[:, -1])
ax.plot(usp2[:, 1], usp2[:, -1])

'''
bar_width = 0.2
ax.bar(range(3), Cifar10[0, :], label='Cifar: Computation time', width=bar_width)
ax.bar(range(3), Cifar10[1, :], label='Cifar: Overhead', width=bar_width, bottom=Cifar10[0, :])
ax.bar(range(3), Cifar10[2, :], label='Cifar: Blocked time', width=bar_width, bottom=Cifar10[0, :] + Cifar10[1, :])

ax.bar(np.array(range(3)) + 1.5 * bar_width, ImageNet[0, :], label='ImageNet: Computation time', width=bar_width)
ax.bar(np.array(range(3)) + 1.5 * bar_width, ImageNet[1, :], label='ImageNet: Overhead', width=bar_width, bottom=ImageNet[0, :])
ax.bar(np.array(range(3)) + 1.5 * bar_width, ImageNet[2, :], label='ImageNet: Blocked time', width=bar_width, bottom=ImageNet[0, :] + ImageNet[1, :])

plt.xticks(np.arange(3)+bar_width/2, ['WK0', 'WK1', 'WK2'])
'''

# plt.title('Time Distribution with STrain')
plt.subplots_adjust(top=0.9, bottom=0.08, left=0.1, right=0.95, hspace=0.4,
                    wspace=0.4)


import os
plt.savefig(os.path.join("fig", os.path.basename(__file__).split(".py")[0] + ".pdf"))


