import matplotlib.pyplot as plt
import numpy as np
import os

from utils import *

def ret_conv_time(a, end, index=0):
	cnt = 0
	for i in range(len(a)):
		if(a[i, 3] < end):
			cnt += 1
		if(cnt >= 10):
			if(index < 0):
				return i
			else:
				return a[i, index]
			break
	return 1000000

dir_name = os.getenv("DATA_DIR")

cifar_ssp_s40 = ret_list(dir_name + '/20190304_02/ps_global_loss_ssp.txt')
cifar_strain = ret_list(dir_name + '/20190304_05/ps_global_loss_usp.txt')
# cifar_strain = ret_list(dir_name + '/20190314_01/ps_global_loss_usp.txt')
cifar_ada = ret_list(dir_name + '/20190309_03/ps_global_loss_ada.txt')
cifar_alter_s40 = ret_list(dir_name + '/20190306_02/ps_global_loss_ssp.txt')
cifar_bsp = ret_list(dir_name + '/20190306_08/ps_global_loss_ssp.txt')

allList = [cifar_strain, cifar_ssp_s40, cifar_bsp, cifar_ada, cifar_alter_s40]
name_list = ['ADSP', 'SSP', 'BSP', 'ADACOMM', 'Fixed\nADACOMM']
# print(cifar_ssp_s40[:10, 3])
end = 1.1
stop_point = [ret_conv_time(l, end=end) for l in allList]
end = 0.94
stop_point[0] = ret_conv_time(allList[0], end=end)
stop_point[-1] = ret_conv_time(allList[-1], end=end)
print(stop_point)
end = 1.1
stop_index = [ret_conv_time(l, end, -1) for l in allList]
end = 0.94
stop_index[0] = ret_conv_time(allList[0], end=end, index=-1)
stop_index[-1] = ret_conv_time(allList[-1], end=end, index=-1)

speedup = [(i - stop_point[0])/i for i in stop_point]
print(speedup)
end = 0.9
stop_point2 = [ret_conv_time(l, end) for l in allList]

for i in range(4):
	stop_point2[i] = max(0, stop_point2[i] - stop_point[i])

plt.figure(num=4, figsize=(8, 6))


ax = plt.subplot(221)
plt.ylim(0.5, 3.5)
for i in range(len(allList)):
	# if(i > 0 and i < len(allList) - 1):
	# if((i != len(allList) - 1) and (i != 0)):
	ax.plot(allList[i][:stop_index[i], 0], savgol_filter(allList[i][:stop_index[i], 3], 10, 3),
			label=name_list[i],
   			linestyle=linestyle_str[i][1],
        	linewidth=linewidth,)
	# else:
	# 	ax.plot(allList[i][:, 0], allList[i][:, 3],  
	# 		linestyle='-',  
	# 		label=name_list[i])
# ax.bar(
# 	range(2),
# 	[stop_point2[0], stop_point2[-1]],
# 	label='Convergence Time',
# 	width=0.3,
# 	# fc='blue',
# 	tick_label = [name_list[0], name_list[-1]],
# 	)
plt.legend()
plt.xlabel('Wall-clock time (s)\n(a)')
plt.ylabel('Global Loss')

bar_width = 0.3
ax = plt.subplot(222)
ax.bar(
	range(len(name_list)),
	stop_point,
	label='Convergence Time',
	width=bar_width,
	# fc='blue',
	tick_label = name_list,
	)

# ax.bar(
# 	range(len(name_list)),
# 	stop_point2,
# 	label='Convergence Time Till 0.9',
# 	width=bar_width,
# 	bottom=stop_point,
# 	# fc='blue',
# 	tick_label = name_list,
# 	)
# plt.legend()
plt.ylabel('Convergence Time (s)')
plt.xlabel('(b)')
plt.yticks(rotation=90)

ax = plt.subplot(223)
for i in range(len(allList)):
	ax.plot(allList[i][:stop_index[i], 0], allList[i][:stop_index[i], 2], 
		linestyle=linestyle_str[i][1],
        linewidth=linewidth,
		# marker='.', 
		# markeredgecolor='red',
		# markeredgewidth=2, 
		label=name_list[i])
plt.legend(loc=1)
plt.xlabel('Wall-clock time (s)\n(c)')
plt.ylabel('Total # of Step')
plt.yticks(rotation=60)

ax = plt.subplot(224)
plt.ylim(0.5, 3.5)
for i in range(len(allList)):
	ax.plot(allList[i][:, 2], savgol_filter(allList[i][:, 3], 10, 3), 
		linestyle=linestyle_str[i][1],
        linewidth=linewidth, 
		label=name_list[i])
plt.legend()
plt.xlabel('# of Steps \n (d)')
plt.ylabel('Global Loss')

# plt.subplots_adjust(top=0.92, bottom=0.12, left=0.10, right=0.95, hspace=0.28,
#                     wspace=0.2)
plt.tight_layout()
plt.savefig("fig/cifar_rst.pdf")


