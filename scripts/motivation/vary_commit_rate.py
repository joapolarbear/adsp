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

dir_name = os.getenv("DATA_DIR")

l0_25 = ret_list(dir_name + '/20190316_02/ps_global_loss_usp.txt')
l0_5 = ret_list(dir_name + '/20190316_01/ps_global_loss_usp.txt')
l1 = ret_list(dir_name + '/20190313_06/ps_global_loss_usp.txt')
l2 = ret_list(dir_name + '/20190315_02/ps_global_loss_usp.txt')
l3 = ret_list(dir_name + '/20190315_03/ps_global_loss_usp.txt')
l4 = ret_list(dir_name + '/20190315_04/ps_global_loss_usp.txt')
l5 = ret_list(dir_name + '/20190315_05/ps_global_loss_usp.txt')
l6 = ret_list(dir_name + '/20190315_06/ps_global_loss_usp.txt')
l10 = ret_list(dir_name + '/20190315_07/ps_global_loss_usp.txt')

allList = [l0_25, l0_5, l1, l2, l3, l4, l5, l6, l10]
name_list = ['0.25', '0.5', '1', '2', '3', '4', '5', '6', '10']
num_list = [0.25, 0.5, 1, 2, 3, 4, 5, 6, 10]
color_list = ['r', 'orange', 'y', 'g', 'lightblue', 'b', 'purple', 'b', 'grey', 'lightgreen']
end = 1.2
stop_point = [ret_conv_time(l, end) for l in allList]
stop_step = [ret_conv_time(l, end, 2) for l in allList]

plt.figure(num=4, figsize=(5, 3))

# ax = plt.subplot(221)
# plt.ylim(0, 3)
# # ax.xaxis.grid(True, which='major')
# # ax.yaxis.grid(True, which='major')
# for i in range(len(allList)):
# 	ax.plot(allList[i][:, 0], allList[i][:, 3], 
# 		color=color_list[i], 
# 		linestyle='-', 
# 		# marker='.', 
# 		# markeredgecolor='red',
# 		# markeredgewidth=2, 
# 		label=name_list[i])
# plt.legend(title=r'$\Delta C_{target}$', ncol=int(len(name_list)/2), loc=1)
# plt.xlabel('Wall-clock time')
# plt.ylabel('Global Loss')

ax = plt.subplot(111)
plt.ylim(1800, 2200)
ax.plot(num_list, stop_point, 
	color='steelblue', 
	linestyle='-', 
	marker='.', 
	markeredgecolor='steelblue',
	# markeredgewidth=2, 
	label='Convergence Time')
plt.yticks(rotation=60)
plt.xlabel(r'$\Delta C_{target}$', fontsize=12)
plt.ylabel('Convergence Time (s)', fontsize=12)
# plt.legend(loc=9, fontsize=12)


# ax = plt.subplot(122)
# # plt.ylim(0, 3000)
# ax.plot(num_list, stop_step, 
# 	color='steelblue', 
# 	linestyle='-', 
# 	marker='.', 
# 	markeredgecolor='steelblue',
# 	# markeredgewidth=2, 
# 	label='Convergence Steps')
# plt.yticks(rotation=60)
# plt.xlabel(r'$\Delta C_{target}$'+'\n(b)', fontsize=12)
# plt.ylabel('Convergence Step', fontsize=12)
# plt.legend(loc=9, fontsize=12)

plt.subplots_adjust(top=0.92, bottom=0.25, left=0.15, right=0.95, hspace=0.25,
                    wspace=0.23)
plt.savefig("fig/vary_commit_rate.pdf")


