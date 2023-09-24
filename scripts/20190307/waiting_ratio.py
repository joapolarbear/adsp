# plot the waiting time vs computation time

import matplotlib.pyplot as plt

# step, time, [cmp_t, ...], avg_cmp_time, cmp_ratio, waiting_ratio
bsp = [609, 2238, [107.547, 87.621, 309.276]]
ssp_s40 = [603, 1974, [109.949, 89.227, 263.697]]
ada = [1749-1109, 1331-1017, [298.188-176.969, 241.785-176.283, 763.400-521.541]]
alter_s40 = [600, 434.36, [101.312, 82.347, 302.180]]

def gen_avg_cmp_time(l):
	cmp_t = l[-1]
	l.append(sum(cmp_t)/float(len(cmp_t)))

gen_avg_cmp_time(bsp)
gen_avg_cmp_time(ssp_s40)
gen_avg_cmp_time(ada)
gen_avg_cmp_time(alter_s40)

def gen_cmp_ratio(l):
	l.append(l[-1]/l[1])
	l.append(1-l[-1])

gen_cmp_ratio(bsp)
gen_cmp_ratio(ssp_s40)
gen_cmp_ratio(ada)
gen_cmp_ratio(alter_s40)

print(ada)

font_size = 15
bar_width = 0.35
name_list = ['BSP', 'SSP s=40', 'ADACOMM', 'Fixed \nADACOMM']

wariting_time_list = [
	bsp[1]-bsp[3], 
	ssp_s40[1]-ssp_s40[3], 
	ada[1]-ada[3], 
	alter_s40[1]-alter_s40[3]]
cmp_time_list = [bsp[3], ssp_s40[3], ada[3], alter_s40[3]]
waiting_ratio_list = [bsp[-1], ssp_s40[-1], ada[-1], alter_s40[-1]]

plt.bar(
	range(len(name_list)),
	wariting_time_list,
	label='Waiting time',
	# fc='orange',
	width=bar_width,
	)

plt.bar(
	range(len(name_list)),
	cmp_time_list,
	label='Cmp Time',
	width=bar_width,
	bottom=wariting_time_list,
	# fc='blue',
	tick_label = name_list,
	)
plt.legend(fontsize = font_size)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.ylabel("Time to Train 600 Steps", fontsize = font_size)
plt.subplots_adjust(top=0.90, bottom=0.14, left=0.17, right=0.9, hspace=0.25,
                    wspace=0.2)
# plt.grid(True)
plt.savefig("waiting_ratio.png")
# plt.savefig("/Users/hhp/Dropbox/ssh/20190307/waiting_ratio.png")


# import numpy as np
# N = 5
# menMeans = (20, 35, 30, 35, 27)
# womenMeans = (25, 32, 34, 20, 25)
# menStd = (2, 3, 4, 1, 2)
# womenStd = (3, 5, 2, 3, 3)
# ind = np.arange(N)    # the x locations for the groups
# width = 0.35       # the width of the bars: can also be len(x) sequence

# p1 = plt.bar(ind, menMeans, width, yerr=menStd)
# p2 = plt.bar(ind, womenMeans, width,
#              bottom=menMeans, yerr=womenStd)

# plt.ylabel('Scores')
# plt.title('Scores by group and gender')
# plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
# plt.yticks(np.arange(0, 81, 10))
# plt.legend((p1[0], p2[0]), ('Men', 'Women'))

# plt.show()
