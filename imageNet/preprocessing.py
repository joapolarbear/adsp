from PIL import Image
import io
from array import *
import os
import sys, time
import numpy as np

class ShowProcess():
	"""
	class for showing the process bar
	https://blog.csdn.net/u013832707/article/details/73608504 
	"""
	i = 0 # current process
	max_steps = 0 # total number of times needed to be processed
	max_arrow = 50 # the lengh of the process bar
	infoDone = 'done'

	# initial, require to define max_steps
	def __init__(self, max_steps, infoDone = 'Done'):
		self.max_steps = max_steps
		self.i = 0
		self.infoDone = infoDone
		self.t0 = time.time()

	def time_format(self, t):
		h = t // 3600
		m = t % 3600 // 60
		s = t % 3600 % 60
		return '%d:%02d:%02d' % (h, m, s)
	# function of showing the bar according to the current process i
	# [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
	def show_process(self, i=None):
		if i is not None:
		    self.i = i
		else:
		    self.i += 1
		num_arrow = int(self.i * self.max_arrow / self.max_steps) # the number of '>' to show
		num_line = self.max_arrow - num_arrow # the number of '-' to show
		percent = self.i * 100.0 / self.max_steps # calculate the process, the format is xx.xx%
		
		# calculate the time
		time_used = time.time() - self.t0
		pre_time = time_used / (percent / 100)
		remain_time = pre_time - time_used

		process_bar = '[' + '>' * num_arrow + '-' * num_line + ']' \
				+ '%.2f' % percent + '%' \
				+ ' Total time: ' + self.time_format(pre_time) \
				+ ' Used time: ' + self.time_format(time_used) \
				+ ' Remain time: ' + self.time_format(remain_time) \
				+ '\n' # string with the output, '\r' means to go back to the left side of the current row
		sys.stdout.write(process_bar) # write the info the the current row
		sys.stdout.flush()
		if self.i >= self.max_steps:
		    self.close()

	def close(self):
		print('')
		print(self.infoDone)
		self.i = 0

data = array('B') # each element takes 1B
tardir = 'train'

def img2bin(path, label):
	global data
	# for label, use 2 bytes to store the label: high low -> high * 256 + low
	high = label // 256
	low = label % 256
	data.append(high)
	data.append(low)

	# for image

	img = Image.open(path)
	pix = img.resize((274,274)).load()
	# if(isinstance(pix[0, 0], int)):
	# 	print('Image \'%s\' is single-band' % path)
	for color in range(3):
		for x in range(274):
			for y in range(274):
				pixel = pix[x,y]
				if(isinstance(pixel, int)):
					data.append(pixel)
				else:
					data.append(pixel[color])
'''
# read and shuffle filenames, and save
fileque = []
for root, dirnames, filenames in os.walk('./' + tardir, topdown=True):
	# print(root, dirnames, filenames)
	for filename in filenames:
		if filename.endswith('.png') or filename.endswith('.jpg'):
			path = os.path.join(root, filename)
			# print(path)
			fileque.append(path)
print('%d images in total' % (len(fileque))) # 1232167 images in total
import random
random.seed(10)
random.shuffle(fileque)
np.save("fileque.npy",np.array(fileque))
'''

worker_index = 1
fileque = np.load("fileque.npy").tolist()
total_length = len(fileque)
if(worker_index < 3):
	fileque = fileque[worker_index * total_length // 4 : (worker_index+1) * total_length // 4]
else:
	fileque = fileque[worker_index * total_length // 4 :]
print('Worker %d need to process %d images in total' % (worker_index, len(fileque)))
process_bar = ShowProcess(len(fileque))
index = 0
for path in fileque:
	label = int(path.split('/')[-2]) - 1
	# print (label)
	img2bin(path, label)
	index += 1
	if index % 100 == 0:
		process_bar.show_process(index)

print('Now, store the binary')

############################################
#write all to binary, all set for cifar10!!#
#https://github.com/gskielian/PNG-2-CIFAR10/blob/master/convert-images-to-cifar-format.py#
############################################
output_file = open(tardir + '_%d.bin' % worker_index, 'wb')
data.tofile(output_file)
output_file.close()
print('All images have been converted to Bin')
#run: nohup python -u  preprocessing.py > nohup.txt 2>&1 &

