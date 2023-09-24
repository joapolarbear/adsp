import matplotlib.pyplot as plt
import numpy as np

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
		return 10000

dir_name = "/Users/hhp/Desktop/STrainData&Record/resultData/amazon"

rail_ssp_s40 = ret_list(dir_name + '/20190311_02/ps_global_loss_ssp.txt')
rail_strain = ret_list(dir_name + '/20190310_02/ps_global_loss_usp.txt')
rail_ada = ret_list(dir_name + '/20190309_04/ps_global_loss_ada.txt')
rail_alter_s40 = ret_list(dir_name + '/20190309_01/ps_global_loss_ada.txt')
rail_bsp = ret_list(dir_name + '/20190310_03/ps_global_loss_ssp.txt')
allList = [rail_strain, rail_ssp_s40, rail_bsp, rail_ada, rail_alter_s40]
name_list = ['ADSP', 'SSP', 'BSP', 'ADACOMM', 'Fixed\nADACOMM']
end = 2.3
stop_point = [ret_conv_time(l, end) for l in allList]
stop_point[0] = ret_conv_time(rail_strain, 2.2)
stop_point[3] = ret_conv_time(rail_ada, 2.4)
# stop_point = [
# 	ret_conv_time(rail_ssp_s40, end),
# 	ret_conv_time(rail_strain, 2.2),
# 	ret_conv_time(rail_ada, 2.4),
# 	ret_conv_time(rail_alter_s40, end),
# 	ret_conv_time(rail_bsp, end)]
speedup = [(i - stop_point[0])/i for i in stop_point]
print(speedup)
end = 0.9
stop_point2 = [ret_conv_time(l, end) for l in allList]
for i in range(4):
	stop_point2[i] = max(0, stop_point2[i] - stop_point[i])

plt.figure(num=4, figsize=(8, 6))


ax = plt.subplot(221)
plt.ylim(2.1, 2.8)
for i in range(len(allList)):
	ax.plot(allList[i][:, 0], allList[i][:, 3],  
		linestyle='-',  
		label=name_list[i])
plt.legend(ncol=2, loc=2)
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
# plt.legend()
plt.ylabel('Convergence Time (s)')
plt.xlabel('(b)')
plt.yticks(rotation=90)


ax = plt.subplot(223)
for i in range(len(allList)):
	ax.plot(allList[i][:, 0], allList[i][:, 2],  
		linestyle='-',  
		label=name_list[i])
plt.legend(loc=1)
plt.xlabel('Wall-clock time (s)\n(c)')
plt.ylabel('Total # of Steps')
plt.yticks(rotation=60)

ax = plt.subplot(224)
plt.ylim(2.1, 2.8)
for i in range(len(allList)):
	ax.plot(allList[i][:, 2], allList[i][:, 3],  
		linestyle='-',  
		label=name_list[i])
plt.xlabel('# of Steps\n(d)')
plt.xticks(rotation=30)
plt.legend(ncol=2, loc=2)
plt.ylabel('Global Loss')



plt.subplots_adjust(top=0.92, bottom=0.15, left=0.10, right=0.95, hspace=0.3,
                    wspace=0.25)
plt.savefig("fig/rail_rst.pdf")


