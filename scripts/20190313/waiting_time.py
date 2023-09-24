# plot the waiting time vs computation time
# version 2

import matplotlib.pyplot as plt
import re
import numpy as np

style = ['seaborn-dark', 'seaborn-darkgrid', 'seaborn-ticks', 
	'fivethirtyeight', 'seaborn-whitegrid', 'classic', 
	'_classic_test', 'fast', 'seaborn-talk', 'seaborn-dark-palette', # 6 - 10
	'seaborn-bright', 'seaborn-pastel', 'grayscale', # 11 12 13
	'seaborn-notebook', 'ggplot', 'seaborn-colorblind', 
	'seaborn-muted', 'seaborn', 'Solarize_Light2', #17 18 19
	'seaborn-paper', 'bmh', 'tableau-colorblind10', # 20 21 22
	'seaborn-white', 'dark_background', 'seaborn-poster', 'seaborn-deep'] # 23 24 25 26
# plt.style.use(style[-3])

# time, step, [cmp_t, ...], avg_cmp_time, cmp_ratio, waiting_ratio, blocked_time
bsp = [30336, 5420, [885.895, 722.952, 2698.131]] 
ssp_s40 = [16308, 5909, [951.764, 778.537, 2881.260]]
# ada = [3841, 6676, [1074.019, 873.746, 3223.479]]
ada = [3579, 6392, [1020.661, 833.726, 3048.221]]
alter_s40 = [4901.43, 8400, [1277.027, 1042.6046, 4099.428]]
usp = [2448.6, 12709, [2315.087, 2289.489, 2284.228]]



def gen_avg_cmp_time(l):
	cmp_t = l[-1]
	l.append(sum(cmp_t)/float(len(cmp_t)))

gen_avg_cmp_time(bsp)
gen_avg_cmp_time(ssp_s40)
gen_avg_cmp_time(ada)
gen_avg_cmp_time(alter_s40)
gen_avg_cmp_time(usp)

def gen_cmp_ratio(l):
	l.append(l[-1]/l[0])
	l.append(1-l[-1])

gen_cmp_ratio(bsp)
gen_cmp_ratio(ssp_s40)
gen_cmp_ratio(ada)
gen_cmp_ratio(alter_s40)
gen_cmp_ratio(usp)

def get_block_time(file):
	f = open(file, 'r')
	total_blocked_time = 0
	for line in f.readlines():
		rst = re.findall('has been blocked for \d+\.?\d* seconds', line)
		for _rst in rst:
			t = float((re.findall('\d+\.?\d*', _rst))[0])
			# print(t)
			total_blocked_time += t
	return total_blocked_time

dir_name = "/Users/hhp/Desktop/STrainData&Record/resultData/Lab_record"
t = get_block_time(dir_name + '/20190312_07/ps_global_loss_ssp.txt')
t = 13446.33
bsp.append(t)
t = get_block_time(dir_name + '/20190313_02/ps_global_loss_ssp.txt')
t = 6850 # to 1.1
ssp_s40.append(t)
t = 1563.32
ada.append(t)
t = 2109.66667
alter_s40.append(t)
usp.append(0)


# print(ada)

font_size = 12
bar_width = 0.35
name_list = ['BSP', 'SSP s=40', 'ADACOMM', 'ADSP']

def waiting_time_per_step(l):
	return (l[0] - l[3]) / l[1]
def cmp_time_per_step(l):
	return (l[3]) / l[1]

waiting_time_list = [
	bsp[0]-bsp[3], 
	ssp_s40[0]-ssp_s40[3], 
	ada[0]-ada[3], 
	usp[0]-usp[3]]

# waiting_time for each step
waiting_time_per_step_list = [
	waiting_time_per_step(bsp), 
	waiting_time_per_step(ssp_s40), 
	waiting_time_per_step(ada), 
	waiting_time_per_step(usp)]
cmp_time_per_step_list = [
	cmp_time_per_step(bsp), 
	cmp_time_per_step(ssp_s40), 
	cmp_time_per_step(ada), 
	cmp_time_per_step(usp)]

step_list = [bsp[1], ssp_s40[1], ada[1], usp[1]]
cmp_time_list = [bsp[3], ssp_s40[3], ada[3], usp[3]]
waiting_ratio_list = [bsp[5], ssp_s40[5], ada[5], usp[5]]
blocked_time_list = [bsp[6], ssp_s40[6], ada[6], usp[6]]

communicate_time_list = [ x-y for (x, y) in zip(waiting_time_list, blocked_time_list)]




plt.figure(num=4, figsize=(8, 6))

ax = plt.subplot(221)
ax.bar(
	range(len(name_list)),
	cmp_time_list,
	label='Computation Time',
	# fc='orange',
	width=bar_width,
	)

ax.bar(
	range(len(name_list)),
	waiting_time_list,
	label='Waiting Time',
	# fc='orange',
	bottom=cmp_time_list,
	width=bar_width,
	tick_label = name_list,
	)

plt.legend(fontsize = font_size)
# plt.tick_params(axis='both', which='major', labelsize=15)
plt.ylabel("Convergence Time (s)")
plt.xticks(rotation=30)

ax = plt.subplot(222)
ax.bar(
	range(len(name_list)),
	cmp_time_per_step_list,
	label='Computation Time',
	# fc='orange',
	width=bar_width,
	align='center'
	)

ax.bar(
	range(len(name_list)),
	waiting_time_per_step_list,
	label='Waiting Time',
	width=bar_width,
	bottom=cmp_time_per_step_list,
	# fc='blue',
	tick_label = name_list,
	align='center'
	)
plt.legend(fontsize = font_size)
# plt.tick_params(axis='both', which='major', labelsize=15)
plt.ylabel("Convergence Time / # of Steps")
plt.xticks(rotation=30)

ax = plt.subplot(223)
ax.bar(
	range(len(name_list)-2),
	cmp_time_list[2:],
	label='Computation Time',
	# fc='orange',
	width=bar_width,
	align='center'
	)

ax.bar(
	range(len(name_list)-2),
	waiting_time_list[2:],
	label='Waiting Time',
	width=bar_width,
	bottom=cmp_time_list[2:],
	# fc='blue',
	tick_label = name_list[2:],
	align='center'
	)
# plt.legend(fontsize = font_size)
# plt.tick_params(axis='both', which='major', labelsize=15)
plt.ylabel("Convergence Time (s)")
plt.margins(x=bar_width)
# plt.xticks(rotation=30)


ax = plt.subplot(224)
ax.bar(
	range(len(name_list)-2),
	cmp_time_per_step_list[2:],
	label='Computation Time',
	# fc='orange',
	width=bar_width,
	align='center'
	)

ax.bar(
	range(len(name_list)-2),
	waiting_time_per_step_list[2:],
	label='Waiting Time',
	width=bar_width,
	bottom=cmp_time_per_step_list[2:],
	# fc='blue',
	tick_label = name_list[2:],
	align='center'
	)
# plt.legend(fontsize = font_size)
# plt.tick_params(axis='both', which='major', labelsize=15)
plt.ylabel("Convergence Time / # of steps")
# plt.xticks(rotation=30)
plt.margins(x=bar_width)
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9, hspace=0.4,
                    wspace=0.2)
# plt.grid(True)
plt.savefig("waiting_time.pdf")
# plt.savefig("/Users/hhp/Dropbox/ssh/20190307/waiting_ratio.png")

