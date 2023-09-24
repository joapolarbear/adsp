'''
	after change the code to the standard form ==> a package
'''
import matplotlib.pyplot as plt
import numpy as np
import os

from utils import ret_list, ret_accuracy

dir_name = os.getenv("DATA_DIR")

# usp = ret_list(dir_name + '/20190417_01/ps_global_loss_usp.txt')
usp = ret_list(dir_name + '/20190430_03/ps_global_loss_usp.txt')
bsp = ret_list(dir_name + '/20190418_01/ps_global_loss_ssp.txt')
ssp = ret_list(dir_name + '/20190418_02/ps_global_loss_ssp.txt')
fixed_ada = ret_list(dir_name + '/20190420_03/ps_global_loss_ada.txt')
ada = ret_list(dir_name + '/20190422_02/ps_global_loss_ada.txt')
usp_all = ret_list(dir_name + '/20190423_02/ps_global_loss_usp.txt')
usp_all2 = ret_list(dir_name + '/20190430_01/ps_global_loss_usp.txt')
r2sp = ret_list(dir_name + '/20190506_01/ps_global_loss_r2sp.txt')
r2sp2 = ret_list(dir_name + '/20190506_02/ps_global_loss_r2sp.txt')
name_list = ['STrain', 'SSP s=40', 'BSP', 'ADACOMM', '\nFixed ADACOMM\n tau=40', 'STrain+mu+\neta+online', 'STrain+mu+\neta+offline', 'R2SP+BSP', 'R2SP+Fixed\nAda tau=40']
allList = [usp, ssp, bsp, ada, fixed_ada, usp_all, usp_all2, r2sp, r2sp2]

# usp = ret_accuracy(dir_name + '/20190417_01/ps_global_eval_usp.txt')
usp = ret_accuracy(dir_name + '/20190430_03/ps_global_eval_usp.txt')
bsp = ret_accuracy(dir_name + '/20190418_01/ps_global_eval_ssp.txt')
ssp = ret_accuracy(dir_name + '/20190418_02/ps_global_eval_ssp.txt')
fixed_ada = ret_accuracy(dir_name + '/20190420_03/ps_global_eval_ada.txt')
ada = ret_accuracy(dir_name + '/20190422_02/ps_global_eval_ada.txt')
usp_all = ret_accuracy(dir_name + '/20190423_02/ps_global_eval_usp.txt')
usp_all2_2 = ret_accuracy(dir_name + '/20190430_01/ps_global_eval_usp.txt')
r2sp = ret_accuracy(dir_name + '/20190506_01/ps_global_eval_r2sp.txt')
r2sp2 = ret_accuracy(dir_name + '/20190506_02/ps_global_eval_r2sp.txt')
accuracyList = [usp, ssp, bsp, ada, fixed_ada, usp_all, usp_all2_2, r2sp, r2sp2]

plt.figure(num=4, figsize=(8, 8))

ax = plt.subplot(511)
for i in range(len(allList)):
	ax.plot(allList[i][:, 0], allList[i][:, -1],
		label=name_list[i])

usp_all2_searchtime = [[2.922, 4779.682], [5801.2796, 6982.675], [1000000, 1000000]]
# usp_all2_searchtime = [[15.94, 3630.282], [4648.93, 5863.823], [1000000, 1000000]]
usp_all2_nowait = []
for data in usp_all2:
	s_time = 0
	for searchtime in usp_all2_searchtime:
		if(data[0] >= searchtime[0] and data[0] <= searchtime[1]):
			break
		elif(data[0] > searchtime[1]):
			s_time += searchtime[1] - searchtime[0]
		elif(data[0] < searchtime[0]):
			usp_all2_nowait.append([data[0] - s_time, data[1], data[2], data[3]])
			break
usp_all2_nowait = np.array(usp_all2_nowait)
ax.plot(usp_all2_nowait[:, 0], usp_all2_nowait[:, -1], label='STrain+mu+eta+offline\n(exclude the search time')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
plt.legend(ncol=4, bbox_to_anchor=(0, 1.7), loc='center left')
plt.ylabel('Gloabl Loss')
plt.xlabel('Time (s)')

ax = plt.subplot(512)
bar_width = 0.3
converge_time = [l[-1, 0] for l in allList]
converge_time.append(usp_all2_nowait[-1, 0])
ax.bar(range(len(name_list) + 1), converge_time, width=bar_width, tick_label=name_list + ['STrain+mu+eta+offline\n(exclude the search time'])
plt.ylabel('Convergence Time (s)')
plt.xticks(fontsize=6)

ax = plt.subplot(513)
final_loss = [l[-1, -1] for l in allList]
ax.bar(range(len(name_list)), final_loss, width=bar_width, tick_label=name_list)
plt.ylabel('Final Loss')
plt.xticks(fontsize=6)

ax = plt.subplot(514)
converge_step = [l[-1, 2] for l in allList]
ax.bar(range(len(name_list)), converge_step, width=bar_width, tick_label=name_list)
plt.ylabel('Convergence Step')
plt.xticks(fontsize=6)

ax = plt.subplot(515)
ax.bar(range(len(name_list)), accuracyList, width=bar_width, tick_label=name_list)
plt.ylabel('Accuracy')
plt.yticks(rotation=60)
plt.xticks(fontsize=6)


plt.subplots_adjust(top=0.85, bottom=0.1, left=0.1, right=0.95, hspace=0.5,
                    wspace=0.3)
import os
plt.savefig(os.path.join("fig", os.path.basename(__file__).split(".py")[0] + ".pdf"))

