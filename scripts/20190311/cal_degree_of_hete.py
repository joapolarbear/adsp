# calculate the degree of heterogeneity

import matplotlib.pyplot as plt
import numpy as np

sleeptimeOfLast7WK = 0

def ret_list(filename):
	lines = [[1.0 / float(x) for x in line.rstrip('\n').split('[')[1].split(']')[0].split(',')] for line in open(filename)]
	return np.array(lines)

def ret_list2(filename):
	lines = np.array([[float(x) for x in line.rstrip('\n').split('[')[1].split(']')[0].split(',')] for line in open(filename)])
	for i in range(len(lines)):
		for j in range(len(lines[i])):
			if(j >= len(lines[i]) - 7):
				lines[i, j] = 1.0 / (sleeptimeOfLast7WK + lines[i, j])
			else:
				lines[i, j] = 1.0 / (lines[i, j])
	return lines

def cal_hete(l):
	return sum(l) / float(min(l) * len(l))

dir_name = "/Users/hhp/Desktop/STrainData&Record/resultData/amazon"
cifar_strain = ret_list2(dir_name + '/20190304_05/ps_hete_usp.txt')

for i in range(len(cifar_strain)):
	print(cal_hete(cifar_strain[i]))

# print(cal_hete(cifar_strain[-1]))


