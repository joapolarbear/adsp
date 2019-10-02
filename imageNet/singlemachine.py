'''
 version 4:
 
'''
# from tensorflow import *  # low version of tensorflow, not work on amazon
from tensorflow.python import *   # tensorflow version 1.10 above 
import tensorflow as tf 
import numpy as np 
import os 
import math
import random
from tensorflow.contrib import *

import copy

from socket import *
# import json  # for transport structured data
from scipy.optimize import curve_fit  # for curve fit

import time
import sys   # sys.getsizeof()
import threading


class STRAIN_WK(object):
	def __init__(self, 
		_worker_index = 0, 
		_check_period = 60.0, 
		_init_base_time_step = 20.0, 
		_max_steps = 1000000, 
		_batch_size = 128, 
		_class_num = 10, # must be given
		_base_dir = None,
		_host = 'localhost',
		_port_base = 14200,
		_s = None):
		pass	

	def log(self, s):
		print('%f: %s' % (time.time() + self.time_align - self.timer_from_PS, s))

	def default_func():
		'''
			default fetch data function, user should define their own fetch data function
			train_images, train_labels, eval_images, image_labels, test_images, test_labels = self.func_fetch_data(
				Train = False, 
				Eval = False, 
				Test = True, 
				batch_size = self.final_batch_size)
			* batch_size is optional, if defined, each time fetch data, reshape the input nodes
		'''
		pass

	def register(self, 
		_session = None, 
		_func_fetch_data = default_func,
		_merged = None,
		_train_op = None,
		_loss = None,
		_global_step = None,
		_feed_X = None,
		_feed_Y = None,
		_feed_prediction = None,
		_top_k_op = None,
		_beta = None
		):
		self.func_fetch_data = _func_fetch_data
		self.sess = _session

		self.merged =  _merged
		self.train_op = _train_op
		self.loss = _loss
		self.global_step = _global_step
		self.X = _feed_X
		self.Y = _feed_Y
		self.feed_prediction = _feed_prediction # addional input
		self.top_k_op = _top_k_op

		# self.beta = _beta

	

	def run(self, simulate = 0):
		pass

	

#################################################
#         Parameter server						#
#################################################
# global variables for parameter server


# STrain configuration
START_TIME = time.time()

# global update eqation
mu = 0.6
eta = 1 / 18.0


CHECKPOINT_PER_EPOCH = 1000 # the number of checkpoints in one epoch
WINDOW_LENGHTH_OF_LOSS = 10 # the length of windows for the global loss, used to check whether converge
# CONVERGE_VAR = 0.001 # when the variance of loss in WINDOW_LENGHTH_OF_LOSS less equal to CONVERGE_VAR => converge
CONVERGE_VAR = 0

def average(seq): 
	return float(sum(seq)) / len(seq)

class LossQueue(object):
	'''
		we do not need a get function in this implementation of queue
	'''
	def __init__(self, maxsize = 10):
		self.maxsize = maxsize
		self.queue = [0.0] * self.maxsize
		self.start_point = 0
		self.end_point = 0
		self.cur_size = 0
	def put(self, elem):
		'''
			put elem to the queue
			1. if the queue is not full, direct put the elem in
			2. if the queue is full, keep the size the same and put the elem iteratively
		'''
		if(self.cur_size == 0):
			self.queue[self.end_point] = elem
			self.cur_size += 1
		elif(self.cur_size < self.maxsize):
			self.end_point += 1
			self.queue[self.end_point] = elem
			self.cur_size += 1
		elif(self.end_point < self.maxsize - 1):
			self.end_point += 1
			self.queue[self.end_point] = elem
		else:
			self.end_point = 0
			self.queue[self.end_point] = elem
	def isConverge(self):
		if(self.cur_size < self.maxsize):
			return False, None
		var = np.var(np.array(self.queue))
		if(var < CONVERGE_VAR): 
			return True, var
		else: 
			return False, var

class STRAIN_PS(object):
	def __init__(self,
		_total_worker_num = 3,
		_check_period = 60.0, 
		_class_num = 10,
		_base_dir = './',
		_host = 'localhost',
		_port_base = 14200,
		_band_width_limit = None,
		_training_end = 0.0,
		_epsilon = 0.3,
		_batch_size = None,
		_s = None
		):

		self.base_dir = _base_dir
		self.max_steps = 1000000
		# record to files
		self.f_log = open(os.path.join(self.base_dir + 'ps_log_usp.txt'), 'w')

	def default_func():
		'''
			default fetch data function, user should define their own fetch data function
			train_images, train_labels, eval_images, image_labels, test_images, test_labels = self.func_fetch_data(
				Train = False, 
				Eval = False, 
				Test = True, 
				batch_size = self.final_batch_size)
			* batch_size is optional, if defined, each time fetch data, reshape the input nodes
		'''
		pass

	def register(self, 
		_session = None, 
		_func_fetch_data = default_func,
		_merged = None,
		_train_op = None,
		_loss = None,
		_global_step = None,
		_feed_X = None,
		_feed_Y = None,
		_feed_prediction = None,
		_top_k_op = None,
		_beta = None
		):
		self.func_fetch_data = _func_fetch_data
		self.sess = _session

		self.merged =  _merged
		self.train_op = _train_op
		self.loss = _loss
		self.global_step = _global_step
		self.X = _feed_X
		self.Y = _feed_Y
		self.feed_prediction = _feed_prediction
		self.top_k_op = _top_k_op

	def log(self, s):
		global START_TIME
		print('%f: %s' % (time.time() - START_TIME, s))

	# main run of the ps scheduler
	def run(self):
		ep = 0
		# while not self.sess.should_stop():
		while (ep < self.max_steps):
			time_step_start = time.time()
			ep += 1

			train_X, train_Y, eval_X, eval_Y, test_X, test_Y = self.func_fetch_data(Train = True, Eval = True, Test = True)
			# train one mini batch
			tmp_time = time.time()
			_, cur_loss, cur_step, top_k_op = \
				self.sess.run([self.train_op, self.loss, self.global_step, self.top_k_op], 
				feed_dict = {self.X: train_X, self.Y: train_Y})
			hete_time = time.time() - tmp_time # record the time to calulate one mini-batch only
			
			# record infomation to log file
			global START_TIME
			cur_time = time.time() - START_TIME

			self.log('Step:%d  Time:%f Train loss:%.20f  result: %s'	% (ep, cur_time, cur_loss, str(top_k_op)))
			self.log('label: %s' %(str(train_Y)))

			# record the a time	
			time_one_step = time.time() - time_step_start # record the time one step needs, excluding the sleep time 			
			# self.log("Time per batch: %f" % (hete_time))
			# self.log("Time one step: %f" % (time_one_step))

		self.log('Stop training - Step:%d\tTime:%f' % (ep, cur_time))





