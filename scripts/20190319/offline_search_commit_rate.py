import matplotlib.pyplot as plt
import numpy as np
import copy

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

data_dir = "/Users/hhp/Desktop/STrainData&Record/resultData/Lab_record"
# data_dir = 'C:/Users/Admin/Dropbox/Lab_record/new/Lab_record/'
strain = ret_list(data_dir + '/20190122_01/ps_global_loss_usp2.txt')
# cifar_strain = ret_list('/Users/hhp/Desktop/amazon/20190314_01/ps_global_loss_usp.txt')
alter = ret_list(data_dir + '/20190111_01/ps_global_loss_ssp2.txt')
strain_plus = ret_list(data_dir + '/20190218_01/ps_global_loss_usp.txt')
strain_plus_nosearch = []
searchTime = [[97.47, 6113.06], [6871.09, 12747.90], [13533.26, 19269.63], [20033.92, 36913.39], [37703.89, 43351.53], [44129.81, 58031.12], [58902.05, 64244.98], [65090.68, 70825.47], [71647.39, 10000000]]
for i in range(len(strain_plus)):
	tmp = 0
	c = copy.deepcopy(strain_plus[i])
	for j in range(len(searchTime)):
		if (strain_plus[i, 0] <= searchTime[j][0]):
			strain_plus_nosearch.append(c)
			strain_plus_nosearch[-1][0] -= tmp
			break
		elif(strain_plus[i, 0] <= searchTime[j][1]):
			break
		else:
			tmp += searchTime[j][1] - searchTime[j][0]
strain_plus_nosearch = np.array(strain_plus_nosearch)


allList = [strain, alter, strain_plus, strain_plus_nosearch]
name_list = ['ADSP', 'Fixed ADACOMM', 'ADSP+', 'ADSP+ \n(excluding the search time)']
end = 1.0
stop_point = [ret_conv_time(l, end) for l in allList]
stop_step = [ret_conv_time(l, end, 2) for l in allList]
stepPerTime = [s / t for (s, t) in zip(stop_step, stop_point)]

plt.figure(num=4, figsize=(5, 3))

# ax = plt.subplot(121)
# ax.plot(strain_plus[:, 0], strain_plus[:, 3], label='STrain Plus')
# ax.plot(strain_plus_nosearch[:, 0], strain_plus_nosearch[:, 3], label=name_list[-1])
# plt.legend()
# plt.xlabel('Wall-clock time (s) \n(a)')
# plt.ylabel('Global Loss')
# # plt.ylim(0, 3)
# plt.xlim(0, 25000)


ax = plt.subplot(111)
for i in range(len(allList)):
	ax.plot(allList[i][:, 0], allList[i][:, 3], 
		# marker='.', 
		# markeredgecolor='red',
		# markeredgewidth=2, 
		label=name_list[i])
plt.legend(loc=1)
plt.xlabel('Wall-clock time (s)')
plt.ylabel('Global Loss')
plt.ylim(0, 8)
plt.xlim(0, 25000)

# plt.xticks(rotation=60)

plt.subplots_adjust(top=0.92, bottom=0.2, left=0.10, right=0.95, hspace=0.2,
                    wspace=0.25)
plt.savefig("fig/offline_search_commit_rate.pdf")


