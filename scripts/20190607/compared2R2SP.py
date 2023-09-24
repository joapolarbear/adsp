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

bsp = ret_list(data_dir + '/20190312_07/ps_global_loss_ssp.txt')
ssp_40 = ret_list(data_dir + '/20190313_02/ps_global_loss_ssp.txt')
# ada = ret_list(data_dir + '/20190313_03/ps_global_loss_ada.txt')
ada = ret_list(data_dir + '/20190409_02/ps_global_loss_ada.txt')
# alter_40 = ret_list(data_dir + '/20190313_05/ps_global_loss_ssp.txt')
alter_40 = ret_list(data_dir + '/20190401_01/ps_global_loss_ssp.txt')
strain_3 = ret_list(data_dir + '/20190327_01/ps_global_loss_usp.txt')
r2sp = ret_list(data_dir + '/20190325_01/ps_global_loss_usp.txt')
r2sp_1 = ret_list(data_dir + '/20190401_02/ps_global_loss_usp.txt')

allList = [bsp, ssp_40, ada, alter_40, strain_3, r2sp, r2sp_1]
name_list = ['BSP', 'SSP s=40', 'ADACOMM', 'Fixed ADACOMM', 'ADSP', 'BatchTune BSP', 'BatchTune Fixed\nADACOMM']
end = 1.1
stop_point = [ret_conv_time(l, end) for l in allList]
stop_step = [ret_conv_time(l, end, 2) for l in allList]
stop_point2 = [ret_conv_time(l, 1.0) for l in allList]
stepPerTime = [s / t for (s, t) in zip(stop_step, stop_point)]

plt.figure(num=4, figsize=(8, 4))

# ax = plt.subplot(121)
# ax.plot(strain_plus[:, 0], strain_plus[:, 3], label='STrain Plus')
# ax.plot(strain_plus_nosearch[:, 0], strain_plus_nosearch[:, 3], label=name_list[-1])
# plt.legend()
# plt.xlabel('Wall-clock time (s) \n(a)')
# plt.ylabel('Global Loss')
# # plt.ylim(0, 3)
# plt.xlim(0, 25000)
ax = plt.subplot(121)
for i in range(len(allList)):
	ax.plot(allList[i][:, 0], allList[i][:, 3], 
		# marker='.', 
		# markeredgecolor='red',
		# markeredgewidth=2, 
		label=name_list[i])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
plt.legend(ncol=4, bbox_to_anchor=(0, 1.18), loc='center left')
plt.xlabel('Wall-clock time (s)\n(a)')
plt.ylabel('Global Loss')
plt.ylim(0.5, 3)
plt.xlim(0, 25000)

name_list2 = ['BSP', 'SSP s=40', 'ADACOMM', 'Fixed\nADACOMM', 'STrain', 'BatchTune\nBSP', 'BatchTune\nFixed\nADACOMM']
bar_width = 0.3
ax = plt.subplot(122)
ax.bar(
	range(len(name_list)),
	stop_point,
	label='Convergence time',
	# fc='orange',
	width=bar_width,
	tick_label = name_list2,
	)
plt.ylabel('Convergence Time(s)')
plt.xticks(rotation=45, fontsize=8)
plt.xlabel('(b)')

# ax = plt.subplot(223)
# ax.bar(
# 	range(len(name_list)),
# 	stop_point2,
# 	label='Convergence time',
# 	# fc='orange',
# 	width=bar_width,
# \
# 	tick_label = name_list,
# 	)
# plt.ylabel('Convergence Time Until 1.0')
# plt.xticks(rotation=90, fontsize=8)
# # plt.xticks(rotation=60)
# ax = plt.subplot(224)
# ax.bar(
# 	range(len(name_list)),
# 	[0.4189, 0.7040, 0.063, 0.096, 0.118, 0.4189, 0.096],
# 	label='Bandwidth (Mb/s)',
# 	# fc='orange',
# 	width=bar_width,
# 	tick_label = name_list,
# 	)
# plt.ylabel('Bandwidth (Mb/s)')
# plt.xticks(rotation=90, fontsize=8)

plt.subplots_adjust(top=0.78, bottom=0.18, left=0.08, right=0.95, hspace=0.45,
                    wspace=0.35)
plt.savefig("fig/compared2R2SP.pdf")()


