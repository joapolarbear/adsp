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
		return -1

dir_name = "/Users/hhp/Desktop/STrainData&Record/resultData/amazon"

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

plt.figure(num=4, figsize=(8, 3))
ax = plt.subplot(121)
for i in range(len(allList)):
	ax.plot(allList[i][:, 0], allList[i][:, 3], 
		# marker='.', 
		# markeredgecolor='red',
		# markeredgewidth=2, 
		label=name_list[i])
	# ax.plot(allList[4+i][:, 0], allList[4+i][:, 3], 
	# 	# color=color_list[i], 
	# 	linestyle='-', 
	# 	# marker='.', 
	# 	# markeredgecolor='red',
	# 	# markeredgewidth=2, 
	# 	label=name_list[4+i])
plt.legend(loc=1)
plt.xlabel('Wall-clock time (s) \n(a)')
plt.ylabel('Global Loss')
# plt.ylim(0, 100000)
plt.xlim(0, 15000)


bar_width = 0.2
ax = plt.subplot(122)
# index = np.arange(int(len(name_list)/2))
index = [0, 1]
index2 = [e + 1.6 * bar_width for e in index]
index3 = (np.array(index) + np.array(index2))/2
ax.bar(index, stop_point[:2], 
	width=bar_width,
	# edgecolor="black",
	# markeredgewidth=2
	label = 'Fixed ADACOMM'
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
plt.yticks(rotation=60)
plt.xlabel('(b)')
plt.ylim(0, 16000)
plt.legend(loc=2, ncol=2)
# plt.xticks(rotation=60)


plt.subplots_adjust(top=0.92, bottom=0.2, left=0.10, right=0.95, hspace=0.2,
                    wspace=0.25)
plt.savefig("fig/vary_scale_size.pdf")


