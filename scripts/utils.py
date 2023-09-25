import os
import numpy as np
from scipy.signal import savgol_filter
import seaborn as sns
sns.set_theme(style="whitegrid", color_codes=True)
tips = sns.load_dataset("tips")

marks = ['/', 'x']

linemarks = ["o", '*', 'X', 'd', 'v', 's', 'p', '^', 'h', 'P']
marksize = 4

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
linewidth = 1.5
linestyle_str = [
    # Named
	('solid', 'solid'),      # Same as (0, ()) or '-'
	('dashed', 'dashed'),    # Same as '--'
 	('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
	('dashdot', 'dashdot'),  # Same as '-.'

	# Parameterized
 	('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
  	('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
	# ('loosely dotted',        (0, (1, 10))),
	# ('dotted',                (0, (1, 1))),
	# ('densely dotted',        (0, (1, 1))),
	('long dash with offset', (5, (10, 3))),
	# ('loosely dashed',        (0, (5, 10))),
	# ('dashed',                (0, (5, 5))),
	('densely dashed',        (0, (5, 1))),

	# ('loosely dashdotted',    (0, (3, 10, 1, 10))),
	('dashdotted',            (0, (3, 5, 1, 5))),
	# ('densely dashdotted',    (0, (3, 1, 1, 1))),

	
	('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
 ]


def ret_list(filename):
	lines = [[float(x) for x in line.rstrip('\n').split(',')] for line in open(filename)]
	return np.array(lines)

def ret_accuracy(filename):
	lines = [line.rstrip('\n') for line in open(filename)]
	accuracy = lines[-1].split(',')[-1].split(']')[0]
	return float(accuracy)

def ret_some_time(filename):
	lines = [line.rstrip('\n') for line in open(filename)]
	time = lines[-1].split(':')[-1].split('[')[-1].split(']')[0].split(',')
	return sum([float(_x) for _x in time])/3.0

def ret_some_time2(filename):
	lines = [line.rstrip('\n') for line in open(filename)]
	tmp = lines[-1].split(':')
	time = tmp[-1].split('[')[-1].split(']')[0].split(',')
	return sum([float(_x) for _x in time])/3.0, float(tmp[0])

def normalize(l1, l2, l3, time):
	''' normalize the computation time, commit overhead and blocked time with the total time '''
	a = np.array([l1, l2, l3])
	return a / time

def ret_time_distribution(dirname):
	if(os.path.isfile(dirname + 'ps_cmp_time_usp.txt')):
		cmp_time, total_time = ret_some_time2(dirname + 'ps_cmp_time_usp.txt')
		blocked_time, _ = ret_some_time2(dirname + 'ps_blocked_time_usp.txt')
	elif(os.path.isfile(dirname + 'ps_cmp_time_ssp.txt')):
		cmp_time, total_time = ret_some_time2(dirname + 'ps_cmp_time_ssp.txt')
		blocked_time, _ = ret_some_time2(dirname + 'ps_blocked_time_ssp.txt')
	elif(os.path.isfile(dirname + 'ps_cmp_time_ada.txt')):
		cmp_time, total_time = ret_some_time2(dirname + 'ps_cmp_time_ada.txt')
		blocked_time, _ = ret_some_time2(dirname + 'ps_blocked_time_ada.txt')
	else:
		print('Error: ' + dirname)
		return
	# print(type(total_time), type(cmp_time), type(blocked_time))
	commit_overhead = total_time - cmp_time - blocked_time
	return normalize(cmp_time, blocked_time, commit_overhead, total_time)
