import matplotlib.pyplot as plt
import numpy as np
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
		return 10000

dir_name = os.getenv("DATA_DIR")
# mu0 = ret_list('/Users/hhp/Desktop/Lab_record/20190313_06/ps_global_loss_usp.txt')
mu0 = ret_list(dir_name + '/20190317_03/ps_global_loss_usp.txt')
mu0_025 = ret_list(dir_name + '/20190317_05/ps_global_loss_usp.txt')
mu0_05 = ret_list(dir_name + '/20190317_01/ps_global_loss_usp.txt')
mu0_1 = ret_list(dir_name + '/20190316_03/ps_global_loss_usp.txt')
mu0_15 = ret_list(dir_name + '/20190317_02/ps_global_loss_usp.txt')
mu0_2 = ret_list(dir_name + '/20190316_04/ps_global_loss_usp.txt')
mu0_25 = ret_list(dir_name + '/20190317_04/ps_global_loss_usp.txt')
mu0_3 = ret_list(dir_name + '/20190316_05/ps_global_loss_usp.txt')
mu0_4 = ret_list(dir_name + '/20190316_08/ps_global_loss_usp.txt')
mu0_6 = ret_list(dir_name + '/20190316_06/ps_global_loss_usp.txt')
mu0_9 = ret_list(dir_name + '/20190316_07/ps_global_loss_usp.txt')

allList = [mu0, mu0_025, mu0_05, mu0_1, mu0_15, mu0_2, mu0_25, mu0_3, mu0_4, mu0_6, mu0_9]
name_list = ['0', '0.025', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.4', '0.6', '0.9']
num_list = [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.9]
color_list = ['r', 'orange', 'y', 'g', 'lightblue', 'b', 'purple', 'black', 'grey', 'lightgreen']
end = 1.2
stop_point = [ret_conv_time(l, end) for l in allList]
stop_step = [ret_conv_time(l, end, 2) for l in allList]

plt.figure(num=4, figsize=(8, 3))
# ax = plt.subplot(221)
# plt.ylim(0, 10)
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
# plt.legend(title=r'$\mu$', loc=1)
# plt.xlabel('Wall-clock time')
# plt.ylabel('Global Loss')

# ax = plt.subplot(222)
# plt.ylim(0, 3000)
# ax.plot(num_list, stop_point, 
# 	color='r', 
# 	linestyle='-', 
# 	# marker='.', 
# 	# markeredgecolor='red',
# 	# markeredgewidth=2, 
# 	label='Convergence Time')
# plt.yticks(rotation=60)
# plt.xlabel(r'$\mu$')
# plt.ylabel('Convergence Time')
# plt.legend()
# # plt.xticks(rotation=60)
def func(x):
	k = float(x)
	m = 18.0
	return 1 - 1 / (1 + (1 - 1.0 / m) * m * (0.05/ k) )

k = [(x+1) for x in range(10)]
mu = [func(_k) for _k in k]
opt = [0.15 for _ in k]


ax = plt.subplot(121)
# plt.xlim(0, 1)
plt.ylabel("Momentum", fontsize = 12)
plt.xlabel(r"$\Delta C_{target}$"+'\n(a)', fontsize = 12)
plt.plot(k, mu, 
	color='steelblue', 
	linestyle='-', 
	marker='.', 
	markeredgecolor='steelblue',
	markeredgewidth=2, 
	label=r'$\mu_{implicit} = 1-p$')
plt.plot(k, opt,
	color='black',
	linestyle='-',
	label='optimal momentum'
	)
# x = [1, 2, 3, 4, 4.5]
# xx = [func(_k) for _k in x]
# plt.fill_between(x, xx, opt[:5], color='orange', alpha=0.25)
plt.legend(loc=1, fontsize=12)
plt.tick_params(axis='both', which='major')

ax = plt.subplot(122)
plt.ylim(0, 3000)
# plt.xlim(0, 0.6)
ax.bar(range(len(name_list)), stop_point, 
	color='steelblue',
	width=0.3,
	# markeredgewidth=2, 
	tick_label = name_list,)
plt.yticks(rotation=60)
plt.xlabel(r'$\mu_{implicit}$'+'\n(b)', fontsize=12)
plt.ylabel('Convergence Time (s)', fontsize=12)
plt.xticks(rotation=30)
# plt.legend(loc=4, fontsize=12)


plt.subplots_adjust(top=0.92, bottom=0.3, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.23)
plt.savefig("fig/vary_mu.pdf")


