import matplotlib.pyplot as plt
import numpy as np
import copy
import os

from utils import ret_list

def ret_conv_time(a, end, index=0):
	cnt = 0
	point = None
	for i in range(len(a)):
		if(a[i, 3] < end):
			cnt += 1
		if(cnt >= 10):
			point = a[i, index]
			break

	if(point):
		return point
	else:
		return 0


data_dir = os.getenv("DATA_DIR")


plt.figure(num=4, figsize=(8, 4))
bar_width = 0.3
name_list = ['BSP', 'SSP s=40', 'ADACOMM', 'Fixed\nADACOMM', 'ADSP']
ax = plt.subplot(121)
ax.bar(
	range(len(name_list)),
	[0.4189, 0.7040, 0.063, 0.096, 0.118],
	label='Bandwidth (Mb/s)',
	# fc='orange',
	width=bar_width,
	tick_label = name_list,
	)
plt.ylabel('Bandwidth (Mb/s)')
plt.xticks(rotation=45, fontsize=8)
plt.xlabel('(a)')




usp = ret_list(data_dir + '/20190430_03/ps_global_loss_usp.txt')
bsp = ret_list(data_dir + '/20190418_01/ps_global_loss_ssp.txt')
ssp = ret_list(data_dir + '/20190418_02/ps_global_loss_ssp.txt')
fixed_ada = ret_list(data_dir + '/20190420_03/ps_global_loss_ada.txt')
usp_all2 = ret_list(data_dir + '/20190430_01/ps_global_loss_usp.txt')
name_list = ['ADSP', 'SSP s=40', 'BSP', 'Fixed ADACOMM tau=40', 'ADSP++']
allList = [usp, ssp, bsp, fixed_ada, usp_all2]
ax = plt.subplot(122)
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
ax.plot(usp_all2_nowait[:, 0], usp_all2_nowait[:, -1], label='ADSP++(excluding\nthe search time)')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
plt.legend(ncol=4, bbox_to_anchor=(-1.38, 1.23), loc='center left')
plt.ylabel('Gloabl Loss')
plt.xlabel('Time (s)\n(b)')

plt.subplots_adjust(top=0.72, bottom=0.18, left=0.1, right=0.95, hspace=0.45,
                    wspace=0.35)
plt.savefig(os.path.join("fig", "bandwidth_offline_search_hypeparameter.pdf"))


