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
		return -1

dir_name = "/Users/hhp/Desktop/STrainData&Record/resultData/amazon"

hete1_45_alter = ret_list(dir_name + '/20190306_02/ps_global_loss_ssp.txt')
hete2_0_alter = ret_list(dir_name + '/20190312_05/ps_global_loss_ssp.txt')
hete2_6_alter = ret_list(dir_name + '/20190311_03/ps_global_loss_ssp.txt')
hete3_2_alter = ret_list(dir_name + '/20190312_08/ps_global_loss_ssp.txt')

hete1_45_usp = ret_list(dir_name + '/20190314_01/ps_global_loss_usp.txt')
hete2_0_usp = ret_list(dir_name + '/20190313_08/ps_global_loss_usp.txt')
hete2_6_usp = ret_list(dir_name + '/20190313_10/ps_global_loss_usp.txt')
hete3_2_usp = ret_list(dir_name + '/20190313_04/ps_global_loss_usp.txt')

allList = [hete1_45_alter, hete2_0_alter, hete2_6_alter, hete3_2_alter, hete1_45_usp, hete2_0_usp, hete2_6_usp, hete3_2_usp]
name_list = ['Fixed ADACOMM+hete=1.45', 'Fixed ADACOMM+hete=2.0','Fixed ADACOMM+hete=2.6','Fixed ADACOMM+hete=3.2',
	'ADSP+hete=1.45', 'ADSP+hete=2.0', 'ADSP+hete=2.6', 'ADSP+hete=3.2']
num_list = [1.45, 2, 2.6, 3.2]
color_list = ['r', 'orange', 'y', 'g', 'lightblue', 'b', 'purple', 'b', 'grey', 'lightgreen']
end = 1.0
stop_point = [ret_conv_time(l, end) for l in allList]
stop_step = [ret_conv_time(l, end, 2) for l in allList]
stepPerTime = [s / t for (s, t) in zip(stop_step, stop_point)]
speedUp = [ (stop_point[i] - stop_point[i+4]) / stop_point[i] for i in range(4)]
ticklist=['a', 'b', 'c', 'd']
plt.figure(num=4, figsize=(8, 5))

for i in range(3):
	ax = plt.subplot(231+i)
	ax.plot(allList[i][:, 0], allList[i][:, 3], 
		# color=color_list[i], 
		linestyle='-', 
		# marker='.', 
		# markeredgecolor='red',
		# markeredgewidth=2, 
		label='Fixed ADACOMM')
	ax.plot(allList[4+i][:, 0], allList[4+i][:, 3], 
		# color=color_list[i], 
		linestyle='-', 
		# marker='.', 
		# markeredgecolor='red',
		# markeredgewidth=2, 
		label='ADSP')
	plt.legend()
	plt.xlabel('Wall-clock time (s)\nH=%2.1f\n(%s)' % (num_list[i], ticklist[i]))
	if(i == 0):
		plt.ylabel('Global Loss')

ax = plt.subplot(234)
ax.plot(allList[3][:, 0], allList[3][:, 3], 
	# color=color_list[i], 
	linestyle='-', 
	# marker='.', 
	# markeredgecolor='red',
	# markeredgewidth=2, 
	label='Fixed ADACOMM')
ax.plot(allList[4+3][:, 0], allList[4+3][:, 3], 
	# color=color_list[i], 
	linestyle='-', 
	# marker='.', 
	# markeredgecolor='red',
	# markeredgewidth=2, 
	label='ADSP')
plt.legend()
plt.xlabel('Wall-clock time (s)\nH=%2.1f\n(%s)' % (num_list[3], ticklist[3]))
plt.ylabel('Global Loss')


ax = plt.subplot(235)
bar_width = 0.3
index = np.arange(len(allList)/2)
ax.bar(index, stop_point[:4], width=bar_width, label='Fixed ADACOMM')
ax.bar(index+bar_width, stop_point[4:], width=bar_width, label='ADSP')
plt.xticks(index + bar_width, num_list)
plt.legend(loc=2)
plt.xlabel('Degree of Heterogeneity H\n(e)')
plt.ylabel('Convergence Time (s)')
plt.yticks(rotation=90)
plt.ylim(0, 15000)



###### size
scale_18_alter = ret_list(dir_name + '/20190306_02/ps_global_loss_ssp.txt')
scale_36_alter  = ret_list(dir_name + '/20190314_03/ps_global_loss_ssp.txt')

scale_18_usp = ret_list(dir_name + '/20190314_01/ps_global_loss_usp.txt')
scale_36_usp = ret_list(dir_name + '/20190314_02/ps_global_loss_usp.txt')

allList = [scale_18_alter, scale_36_alter, scale_18_usp, scale_36_usp]
name_list = ['Fixed ADACOMM: 18 workers', 'Fixed ADACOMM: 36 workers','ADSP: 18 workers', 'ADSP: 36 workers']
num_list = [18, 36]
end = 1.0
stop_point = [ret_conv_time(l, end) for l in allList]
stop_step = [ret_conv_time(l, end, 2) for l in allList]
stepPerTime = [s / t for (s, t) in zip(stop_step, stop_point)]
speedUp = [ (stop_point[i] - stop_point[i+int(len(allList)/2)]) / stop_point[i] for i in range(int(len(allList)/2))]
ticklist=['a', 'b', 'c', 'd']

bar_width = 0.2
ax = plt.subplot(236)
# index = np.arange(int(len(name_list)/2))
index = [0, 1]
index2 = [e + 1.6 * bar_width for e in index]
index3 = (np.array(index) + np.array(index2))/2
ax.bar(index, stop_point[:2], 
	width=bar_width,
	# edgecolor="black",
	# markeredgewidth=2
	label = 'Fixed\nADACOMM'
	)
ax.bar(index2, stop_point[2:], 
	width=bar_width,
	# edgecolor="black",
	# markeredgewidth=2, 
	label = 'ADSP'
	)
name_list2=['18 workers', '36 workers']
plt.xticks(index3, name_list2)
plt.ylabel('Convergence Time (s)')
plt.yticks(rotation=90)
plt.xlabel('(f)')
plt.ylim(0, 15000)
plt.legend(loc=2, ncol=1, fontsize=9)
# plt.margins(x=bar_width)


plt.subplots_adjust(top=0.92, bottom=0.15, left=0.10, right=0.95, hspace=0.5,
                    wspace=0.25)
import os
plt.savefig(os.path.join("fig", os.path.basename(__file__).split(".py")[0] + ".pdf"))


