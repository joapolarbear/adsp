import os
import tensorflow as tf
import numpy as np
import random
import sys
import struct

filenames = [os.path.join('./tmp/cifar10_data/cifar-10-batches-bin', 'data_batch_%d.bin' % i)
               for i in xrange(1, 6)]

for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

for f in filenames:
	label_bytes = 1  # 2 for CIFAR-100
	height = 32
	width = 32
	depth = 3
	image_bytes = height * width * depth
	fd = open(f, 'rb')
	fd_w = open(f.split('.bin')[0] + '_ib.bin', 'wb')
	while True:
		save = True
		_byte = fd.read(label_bytes)
		if(not _byte):
			break
		_image = fd.read(image_bytes)
		label, = struct.unpack('B', _byte)
		# print("da:%s %d over" % (str(_byte), label))
		for x in xrange(len(sys.argv) - 1):
			if(label == int(sys.argv[x + 1]) and random.random() < 0.9):
				save = False
		if save:		
			fd_w.write(_byte)
			fd_w.write(_image)



