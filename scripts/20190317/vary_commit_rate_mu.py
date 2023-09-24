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

dir_name = "/Users/hhp/Desktop/STrainData&Record/resultData/Lab_record"


# vary commit_rate first
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

FONT_SIZE = 18
plt.figure(num=4, figsize=(12, 4))


ax = plt.subplot(131)
plt.ylim(1800, 2200)
ax.plot(num_list, np.array(stop_point)/1000, 
	color='steelblue', 
	linestyle='-', 
	marker='.', 
	markeredgecolor='steelblue',
	# markeredgewidth=2, 
	label='Convergence Time')
plt.yticks(fontsize=FONT_SIZE, rotation=0)
plt.xlabel(r'$\Delta C_{target}$' + "\n(a)", fontsize=FONT_SIZE)
plt.ylabel('Convergence \nTime (1000s)', fontsize=FONT_SIZE)
plt.xticks(fontsize=FONT_SIZE)
plt.ylim(1.8, 2.2)

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

def func(x):
	k = float(x)
	m = 18.0
	return 1 - 1 / (1 + (1 - 1.0 / m) * m * (0.05/ k) )

k = [(x+1) for x in range(10)]
mu = [func(_k) for _k in k]
opt = [0.15 for _ in k]


ax = plt.subplot(132)
# plt.xlim(0, 1)
plt.ylabel("Momentum", fontsize = FONT_SIZE)
plt.xlabel(r"$\Delta C_{target}$"+'\n(b)', fontsize = FONT_SIZE)
plt.plot(k, mu, 
	color='steelblue', 
	linestyle='-', 
	marker='^', 
	markeredgecolor='steelblue',
	markeredgewidth=2, 
	label=r'$\mu_{implicit}$')
plt.plot(k, opt,
	color='black',
	linestyle='-',
	label='Opt. '+ r'$\mu_{implicit}$'
	)
# x = [1, 2, 3, 4, 4.5]
# xx = [func(_k) for _k in x]
# plt.fill_between(x, xx, opt[:5], color='orange', alpha=0.25)
# plt.legend(loc=1, fontsize=FONT_SIZE)
plt.tick_params(axis='both', which='major')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
plt.legend(fontsize=FONT_SIZE-1, ncol=1, bbox_to_anchor=(0.11, 0.8), loc='center left')
plt.yticks(fontsize=FONT_SIZE)
plt.xticks(fontsize=FONT_SIZE)

ax = plt.subplot(133)
plt.ylim(0, 3000)
# plt.xlim(0, 0.6)
ax.bar(range(len(name_list)), np.array(stop_point)/1000, 
	color='steelblue',
	width=0.3,
	# markeredgewidth=2, 
	tick_label = name_list,)
plt.yticks(fontsize=FONT_SIZE)
plt.xlabel(r'$\mu_{implicit}$'+' (c)', fontsize=FONT_SIZE)
plt.ylabel('Convergence \nTime (1000s)', fontsize=FONT_SIZE)
plt.xticks(fontsize=FONT_SIZE, rotation=90)
plt.ylim(0, 3)
# plt.legend(loc=4, fontsize=FONT_SIZE)


plt.subplots_adjust(top=0.9, bottom=0.3, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.4)
plt.savefig("fig/vary_commit_rate_mu.pdf")


