import matplotlib.pyplot as plt
import numpy as np
import os

from utils import *

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

dir_name = os.getenv("DATA_DIR")

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
	ax.plot(allList[4+i][:, 0], allList[4+i][:, 3], 
		# color=color_list[i], 
		linestyle=linestyle_str[0][1],
        linewidth=linewidth,
		# marker='.', 
		# markeredgecolor='red',
		# markeredgewidth=2, 
		label='ADSP')
	ax.plot(allList[i][:, 0], allList[i][:, 3], 
		# color=color_list[i], 
		linestyle=linestyle_str[1][1],
        linewidth=linewidth, 
		# marker='.', 
		# markeredgecolor='red',
		# markeredgewidth=2, 
		label='Fixed ADACOMM')
	
	plt.legend()
	plt.xlabel('Wall-clock time (s)\nH=%2.1f\n(%s)' % (num_list[i], ticklist[i]))
	if(i == 0):
		plt.ylabel('Global Loss')

ax = plt.subplot(223)
ax.plot(allList[4+3][:, 0], allList[4+3][:, 3], 
	# color=color_list[i], 
	linestyle=linestyle_str[0][1],
	linewidth=linewidth,
	# marker='.', 
	# markeredgecolor='red',
	# markeredgewidth=2, 
	label='ADSP')
ax.plot(allList[3][:, 0], allList[3][:, 3], 
	# color=color_list[i], 
	linestyle=linestyle_str[1][1],
	linewidth=linewidth,
	# marker='.', 
	# markeredgecolor='red',
	# markeredgewidth=2, 
	label='Fixed ADACOMM')
plt.legend()
plt.xlabel('Wall-clock time (s)\nH=%2.1f\n(%s)' % (num_list[3], ticklist[3]))
plt.ylabel('Global Loss')


ax = plt.subplot(224)
bar_width = 0.3
index = np.arange(len(allList)/2)
bars = ax.bar(index+bar_width, stop_point[4:], width=bar_width, label='ADSP')
for bar in bars:
	bar.set_hatch(marks[0])
bars = ax.bar(index, stop_point[:4], width=bar_width, label='Fixed ADACOMM')
for bar in bars:
	bar.set_hatch(marks[1])

plt.xticks(index + bar_width, num_list)
plt.legend(loc=2)
plt.xlabel('Degree of Heterogeneity H\n(e)')
plt.ylabel('Convergence Time (s)')
plt.yticks(rotation=60)
plt.ylim(0, 15000)

# ax = plt.subplot(223)
# ax.plot(num_list, stop_step[:4], marker='.', label='Alter')
# ax.plot(num_list, stop_step[4:], marker='.', label='ADSP')
# plt.legend()
# plt.xlabel('Degree of Heterogeneity ' + r'$\bar{v} / v_{min}$')
# plt.ylabel('Convergence Step')

# ax = plt.subplot(224)
# ax.plot(num_list, speedUp, marker='.')
# plt.xlabel('Degree of Heterogeneity H\n(f)')
# plt.ylabel('Speed Up ratio')

print (speedUp)

plt.subplots_adjust(top=0.92, bottom=0.18, left=0.10, right=0.95, hspace=0.8,
                    wspace=0.25)
plt.savefig("fig/vary_Hete_degree.pdf")


