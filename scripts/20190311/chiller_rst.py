import matplotlib.pyplot as plt
import numpy as np


def ret_list(filename):
	lines = [[float(x) for x in line.rstrip('\n').split(',')] for line in open(filename)]
	return np.array(lines)
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
		return 10000
dir_name = "/Users/hhp/Desktop/STrainData&Record/resultData/amazon"

rail_ssp_s40 = ret_list(dir_name + '/20190318_01/ps_global_loss_ssp.txt')
rail_strain = ret_list(dir_name + '/20190318_03/ps_global_loss_usp.txt')
rail_ada = ret_list(dir_name + '/20190318_02/ps_global_loss_ada.txt')
rail_alter_s40 = ret_list(dir_name + '/20190318_04/ps_global_loss_ada.txt')
rail_bsp = ret_list(dir_name + '/20190317_07/ps_global_loss_ssp.txt')

allList = [rail_strain, rail_ssp_s40, rail_bsp, rail_ada, rail_alter_s40]
name_list = ['ADSP', 'SSP', 'BSP', 'ADACOMM', 'Fixed\nADACOMM']

end = -22000
stop_point = [ret_conv_time(l, end) for l in allList]
stop_point[3] = ret_conv_time(allList[3], -20000)
stop_step = [ret_conv_time(l, end, 2) for l in allList]

end = 0.9
stop_point2 = [ret_conv_time(l, 0.9) for l in allList]

for i in range(4):
	stop_point2[i] = max(0, stop_point2[i] - stop_point[i])

plt.figure(num=4, figsize=(8, 3))


ax = plt.subplot(121)
for i in range(len(allList)):
	ax.plot(allList[i][:, 0], allList[i][:, 3],
		label=name_list[i])
plt.legend()
plt.xlabel('Wall-clock time (s)\n(a)')
plt.ylabel('Global Loss')
plt.yticks(rotation=60)

# ax = plt.subplot(222)
# for i in range(len(allList)):
# 	ax.plot(allList[i][:, 0], allList[i][:, 2],
# 		label=name_list[i])
# plt.legend()
# plt.xlabel('Wall-clock time (s)')
# plt.ylabel('# of Steps')
# plt.yticks(rotation=90)

# ax = plt.subplot(223)
# plt.ylim(0.5, 3.5)
# for i in range(len(allList)):
# 	ax.plot(allList[i][:, 2], allList[i][:, 3],
# 		label=name_list[i])
# plt.legend()
# plt.xlabel('# of Steps')
# plt.ylabel('Global Loss')

bar_width = 0.3
ax = plt.subplot(122)
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
plt.xticks(rotation=0)
plt.xlabel('(b)')
plt.yticks(rotation=60)

plt.subplots_adjust(top=0.92, bottom=0.25, left=0.10, right=0.95, hspace=0.2,
                    wspace=0.3)
plt.savefig("chiller_rst.pdf")


