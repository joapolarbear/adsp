# plot the waiting time vs computation time
# version 2

import matplotlib.pyplot as plt
import re
import numpy as np

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

def ret_avg_waiting_time(l):
	return (l[0] - l[3]) / l[1]
def ret_avg_cmp_time(l):
	return (l[3]) / l[1]

waiting_time_list1 = [
	bsp[0]-bsp[3], 
	ssp_s40[0]-ssp_s40[3], 
	ada[0]-ada[3], 
	usp[0]-usp[3]]
cmp_time_list1 = [bsp[3], ssp_s40[3], ada[3], usp[3]]
print(cmp_time_list1)
# waiting_time for each step
waiting_time_list2 = [
	ret_avg_waiting_time(bsp), 
	ret_avg_waiting_time(ssp_s40), 
	ret_avg_waiting_time(ada), 
	ret_avg_waiting_time(usp)]
cmp_time_list2 = [
	ret_avg_cmp_time(bsp), 
	ret_avg_cmp_time(ssp_s40), 
	ret_avg_cmp_time(ada), 
	ret_avg_cmp_time(usp)]
waiting_ratio_list = [bsp[5], ssp_s40[5], ada[5], usp[5]]

blocked_time_list = [bsp[6], ssp_s40[6], ada[6], usp[6]]

communicate_time_list = [ x-y for (x, y) in zip(waiting_time_list1, blocked_time_list)]
print(communicate_time_list)
print(blocked_time_list)
step_list = [bsp[1], ssp_s40[1], ada[1], usp[1]]



plt.figure(num=4, figsize=(8, 6))

ax = plt.subplot(221)
ax.bar(
	range(len(name_list)),
	cmp_time_list1,
	label='Cmp time',
	# fc='orange',
	width=bar_width,
	)

ax.bar(
	range(len(name_list)),
	communicate_time_list,
	label='Communication time',
	# fc='orange',
	bottom=cmp_time_list1,
	width=bar_width,
	)

ax.bar(
	range(len(name_list)),
	blocked_time_list,
	label='Blocked Time',
	width=bar_width,
	bottom=np.array(cmp_time_list1) + np.array(communicate_time_list),
	# fc='blue',
	tick_label = name_list,
	)
plt.legend(fontsize = font_size)
# plt.tick_params(axis='both', which='major', labelsize=15)
plt.ylabel("Convergence Time (s)")
plt.xticks(rotation=30)

ax = plt.subplot(222)
ax.bar(
	range(len(name_list)),
	cmp_time_list2,
	label='Waiting time',
	# fc='orange',
	width=bar_width,
	)

ax.bar(
	range(len(name_list)),
	np.array(communicate_time_list)/ np.array(step_list) ,
	label='Cmp Time',
	width=bar_width,
	bottom=cmp_time_list2,
	# fc='blue',
	tick_label = name_list,
	)
ax.bar(
	range(len(name_list)),
	np.array(blocked_time_list) / np.array(step_list),
	label='Blocked Time',
	width=bar_width,
	bottom=np.array(cmp_time_list2) + np.array(communicate_time_list) / np.array(step_list),
	# fc='blue',
	tick_label = name_list,
	)
plt.legend(fontsize = font_size)
# plt.tick_params(axis='both', which='major', labelsize=15)
plt.ylabel("Convergence Time / # of Steps")
plt.xticks(rotation=30)

ax = plt.subplot(223)
ax.bar(
	range(len(name_list)-2),
	waiting_time_list1[2:],
	label='Waiting time',
	# fc='orange',
	width=bar_width,
	)

ax.bar(
	range(len(name_list)-2),
	cmp_time_list1[2:],
	label='Cmp Time',
	width=bar_width,
	bottom=waiting_time_list1[2:],
	# fc='blue',
	tick_label = name_list[2:],
	)
# plt.legend(fontsize = font_size)
# plt.tick_params(axis='both', which='major', labelsize=15)
plt.ylabel("Convergence Time (s)")
# plt.xticks(rotation=30)

ax = plt.subplot(224)
ax.bar(
	range(len(name_list)-2),
	waiting_time_list2[2:],
	label='Waiting time',
	# fc='orange',
	width=bar_width,
	)

ax.bar(
	range(len(name_list)-2),
	cmp_time_list2[2:],
	label='Cmp Time',
	width=bar_width,
	bottom=waiting_time_list2[2:],
	# fc='blue',
	tick_label = name_list[2:],
	)
# plt.legend(fontsize = font_size)
# plt.tick_params(axis='both', which='major', labelsize=15)
plt.ylabel("Convergence Time / # of steps")
# plt.xticks(rotation=30)

plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9, hspace=0.4,
                    wspace=0.2)
# plt.grid(True)
plt.savefig("fig/waiting_time.pdf")
