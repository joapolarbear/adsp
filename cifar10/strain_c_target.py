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

beta = None
MIN_TIME_TO_COMMIT = 10 # avoid frequent commit

COMMUNICATION_SLEEP = 0

#################################################
#         Parameter server						#
#################################################
# global variables for parameter server
worker_num = 3
class_num = 10

expect_commit = []

# information for each worker
worker_step_list = []
worker_loss_list = []
commit_cnt_list = []
original_worker_speed_list = []
worker_speed_list = []
num_of_local_update = [] # the number of local update 
computation_time_list = []
commit_overhead_list = []
blocked_time_list = []

# global parameters 
parameters = []
v_para = []

# STrain configuration
isfirst = True
MOMENTUN = False
START_TIME = time.time()

# global update eqation
mu = 0.6
eta = 1 / 18.0

# define a lock to access global variables
mutex = threading.Lock()
checkInCnt = 0

CHECKPOINT_PER_EPOCH = 1000 # the number of checkpoints in one epoch
WINDOW_LENGHTH_OF_LOSS = 10 # the length of windows for the global loss, used to check whether converge
# CONVERGE_VAR = 0.001 # when the variance of loss in WINDOW_LENGHTH_OF_LOSS less equal to CONVERGE_VAR => converge
CONVERGE_VAR = 0.0


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

		self.No = int(_worker_index)
		self.check_period = float(_check_period)
		self.base_time_step = float(_init_base_time_step) # initial , commit update per 20s
		self.max_steps = int(_max_steps)
		self.batch_size = int(_batch_size)
		self.class_num = int(_class_num) #
		self.base_dir = _base_dir
		self.host = _host
		self.port_base = int(_port_base)

		global beta
		with tf.device('/cpu:0'):
			beta = tf.get_variable(name='beta', shape=[self.class_num], initializer=tf.ones_initializer(), dtype=tf.float32, trainable=False)

        ##########################################################

		self.commit_cnt = 0 # record the total commit number
		self.class_cnt = [0 for _ in xrange(self.class_num)]

		## for prediction
		self.predict_cnt = [0 for _ in xrange(self.class_num)]
		self.predict_rst = [0 for _ in xrange(self.class_num)]
		self.eval_rst = [0.0 for _ in xrange(self.class_num + 1)] # last elem is the overall accuracy

		# log for the worker
		self.f_log = open(os.path.join(self.base_dir + 'wk_%d_usp.txt' % (self.No)), 'w')
		self.f_pre = open(os.path.join(self.base_dir + 'wk_%d_usp_pred.txt' % (self.No)), 'w')

		# store the parameters
		self.parameter = []  # a list of parameters, parameters are np.array
		self.para_shape = []

		self.commit_overhead = 0

	def log(self, s):
		print('%f: %s' % (time.time() + self.time_align - self.timer_from_PS, s))

	def str2list(self, s):
		return [float(x) for x in s.split('[')[1].split(']')[0].split(',')]
	
	def sendmsg(self, s):
		''' send long message, protocol 
			1: send len; 2: recv start; 3: send msg; 4: recv ok		
		'''
		time.sleep(COMMUNICATION_SLEEP)
		tmp_time = time.time()
		length = len(s)
		self.skt.sendall(str(length))
		self.skt.recv(self.buffsize)
		self.skt.sendall(s)
		self.skt.recv(self.buffsize)
		return time.time() - tmp_time

	def recvmsg(self):
		''' recv long message, protocol 
			1: recv len; 2: send start; 3: recv msg; 4: send ok
		'''
		time.sleep(COMMUNICATION_SLEEP)
		s = ''
		length = int(self.skt.recv(self.buffsize))
		self.skt.sendall('Start')
		while(length > 0):
			msg = self.skt.recv(self.buffsize)
			s += msg
			length -= len(msg)
		self.skt.sendall('OK')
		# print s
		return s

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
		
	def connect2PS(self):
		# connect to the PS
		addr = (self.host, self.port_base + self.No)
		self.skt = socket(AF_INET,SOCK_STREAM)
		self.skt.connect(addr)
		self.buffsize = 1024
		msg = self.skt.recv(self.buffsize).split(',')
		self.timer_from_PS = float(msg[0])
		self.time_align = float(msg[1]) - time.time()

		self.skt.sendall("Recv start_t")
		msg = self.skt.recv(self.buffsize)
		if('OK' in msg):
			# if the worker is the first worker to connect PS, upload the parameters
			for v in tf.trainable_variables():
				tmp = v.eval(session = self.sess)
				self.parameter.append(tmp)
				shape = v.get_shape()
				self.para_shape.append(shape)	
				msg = tmp.tobytes()
				self.sendmsg(msg)

				# debug
				self.log('Communication size: %fB' % (sys.getsizeof(msg)))
				self.log('Storage size: %fB' % (sys.getsizeof(tmp)))
				self.log('Storage size: %fB' % (sys.getsizeof(self.parameter[-1])))
				self.log("Shape: %s" % (str(shape)))
				print
	
			self.sendmsg('$')
		else:
			# if the worker is not the first worker to connect PS, load the parameters
			self.skt.sendall('Load')
			# Receive  theoverall parameters from the PS
			for v in tf.trainable_variables():
				msg = self.recvmsg()
				tmp = np.copy(np.frombuffer(msg, dtype=np.float32))
				shape = v.get_shape()
				self.para_shape.append(shape)
				self.parameter.append(tmp.reshape(shape))

				# debug
				self.log('Communication size: %fB' % (sys.getsizeof(msg)))
				self.log('Storage size: %fB' % (sys.getsizeof(tmp)))
				self.log("Shape: %s" % (str(shape)))
				print	
			self.skt.recv(self.buffsize)
			# update the local mode
			index = 0
			for v in tf.trainable_variables():
				v.load(self.parameter[index], self.sess)
				index += 1
		self.log("Worker %d successfully connct to the "
		 	"parameter server [%s:%d]" % (self.No, self.host, (self.port_base + self.No)))

	def run(self, simulate = 0):
		self.connect2PS()
		ep = 0
		time_after_a_commit = time.time()

		si = 0 # the number of local updates
		self.notWait = False
		# while not self.sess.should_stop():
		while (ep < self.max_steps):
			time_step_start = time.time()
			ep += 1
			si += 1
			# if(simulate):
			time.sleep(simulate)
			train_X, train_Y, eval_X, eval_Y, test_X, test_Y = self.func_fetch_data(Train = True, Eval = True, Test = True)
			cur_train_labels_list = train_Y.tolist()
			# train one mini batch
			tmp_time = time.time()
			_, cur_loss, cur_step = \
				self.sess.run([self.train_op, self.loss, self.global_step], 
				feed_dict = {self.X: train_X, self.Y: train_Y})
			hete_time = time.time() - tmp_time # record the time to calulate one mini-batch only
			
			# record infomation to log file
			cur_time = time.time() - self.timer_from_PS
			self.f_log.write('%020.10f, %-10d, %030.20f\n' % (cur_time, ep, cur_loss))
			self.f_log.flush()

			# evaluation
			if(ep % 10 == 0):
				_, eval_loss = \
					self.sess.run([self.train_op, self.loss], 
					feed_dict = {self.X: eval_X, self.Y: eval_Y})
				cur_time = time.time() - self.timer_from_PS	
				self.log('USP_WK%d - Step:%d\tTime:%f\tbase_time_step:%ds\tTrain loss: %f\tEval loss: %f'\
					% (self.No, ep, cur_time, self.base_time_step, cur_loss, eval_loss))
			else:
				self.log('USP_WK%d - Step:%d\tTime:%f\tbase_time_step:%ds\tTrain loss: %f'\
					% (self.No, ep, cur_time, self.base_time_step, cur_loss))

			# record the a time	
			time_one_step = time.time() - time_step_start # record the time one step needs, excluding the sleep time 			
			time_span = time.time() - time_after_a_commit # record the time apart from last commit
			self.log("Time per batch: %f" % (hete_time))
			self.log("Time one step: %f" % (time_one_step))
			self.log("Time from last commit: %f" % (time_span))
			#  commit to the PS 
			if((self.notWait and (time_span >= self.base_time_step)) or ((not self.notWait) and si >= self.base_time_step)):
				tmp_time = time.time()
				self.wk_process_commit(cur_time, cur_loss, hete_time, time_one_step, ep, si, time_span)
				self.commit_overhead_pertime = time.time() - tmp_time
				self.commit_overhead += self.commit_overhead_pertime
				# check whether to check 
				self.skt.sendall('Check?')
				msg = self.skt.recv(self.buffsize)
				if(not self.notWait):
					self.log("Allow waiting, the number of local updates meets the threshold => commit => check ?: %s" % msg)
				else:
					self.log("No waiting, the time meets the threshold => commit => check ?: %s" % msg)
				
				# decide check or not
				if('Check' in msg):
					self.wk_process_check(cur_time, ep)
				elif('Stop' in msg):
					break
				# recalculate time_after_a_commit after one commit, reset the number of local updates
				time_after_a_commit = time.time()
				si = 0
			# end commit
		#end while
		self.log('USP_WK%d stops training - Step:%d\tTime:%f' % (self.No, ep, cur_time))

	def wk_process_commit(self, cur_time, cur_loss, hete_time, time_one_step, cur_step, si, time_span):
		self.commit_cnt += 1
		self.skt.sendall('Commit')
		self.skt.recv(self.buffsize)

		tmp_time = time.time()
		# Sum and send all local updates
		index = 0
		communication_size = 0
		for v in tf.trainable_variables():
			cur_parameter = v.eval(session = self.sess)
			l = (cur_parameter - self.parameter[index]).tobytes()
			communication_size += sys.getsizeof(l)
			self.sendmsg(l)
			self.parameter[index] = cur_parameter[:]
			index += 1
		# end for
		print
		self.log('*************************************** commit *********************************************************')
		self.log("The size of transmitted Parameters: %f M" % (float(communication_size)/(1024.0 * 1024.0)))
		self.log("Time of extracting and sending parameters: %f" % (time.time() - tmp_time))

		tmp_time = time.time()
		## send some trival message
		l = "%20.15f, %20.15f, %d, %20.15f, %d, %20.15f" % (hete_time, cur_loss, cur_step, time_one_step, si, time_span)
		self.skt.sendall(l)
		self.log("The size of (send some trival message): %f M" % (float(sys.getsizeof(l))/(1024.0 * 1024.0)))
		self.log("Time of sending trival messages: %f" % (time.time() - tmp_time))

		tmp_time = time.time()
		##  Receive the overall parameters from the PS
		communication_size = 0
		for i in xrange(len(self.parameter)):
			msg = self.recvmsg()
			communication_size += sys.getsizeof(msg)
			self.parameter[i] = np.copy(np.frombuffer(msg, dtype=np.float32)).reshape(self.para_shape[i])
		self.log("The size of transmitted Parameters: %d M" % (float(communication_size)/(1024.0 * 1024.0)))
		self.log("Time of recv parameters: %f" % (time.time() - tmp_time))

		# receive trival message
		self.skt.recv(self.buffsize)
		
		tmp_time = time.time()
		# Update the local mode
		index = 0
		for v in tf.trainable_variables():
			tt = time.time()
			v.load(self.parameter[index], self.sess)
			# self.sess.run(assign_op)
			self.log("		-- time: %f\tsize: %f" %(time.time() - tt, sys.getsizeof(self.parameter[index])))
			index += 1
		self.log("Time of load parameters: %f" % (time.time() - tmp_time))
		self.log("USP_WK%d: Commit %d finished" % (self.No, self.commit_cnt))
		self.log('******************************************************************************************************')
		print

	def wk_process_check(self, cur_time, cur_step):
		# change the role from sender to receiver
		## send some trival message
		l = "%20.15f, " % (self.commit_overhead)
		self.skt.sendall(l)

		# receive target commit No 
		expect_cNo = float(self.skt.recv(self.buffsize))
		print
		self.log('########################## check #################################')
		self.log("New delta c target: %f" % (expect_cNo))
		
		# update the base_time_step
		if(expect_cNo < 0):
			# based on the time, no waiting
			# huhanpeng: estimate the communication time ???
			self.base_time_step = max(MIN_TIME_TO_COMMIT, self.check_period / float(- expect_cNo) - self.commit_overhead_pertime)
			self.notWait = True
		else:
			self.base_time_step = expect_cNo
			self.notWait = False	
		self.log('#################################################################')
		print


def average(seq): 
	return float(sum(seq)) / len(seq)

class PSThread(threading.Thread):	
	def __init__(self, worker_index, _host, _port_base):
		threading.Thread.__init__(self)
		self.host = _host
	 	self.port_base = _port_base
	 	self.buffsize = 1024
	 	self.index = worker_index
	 	self.isCheckOrStop = 0	# 0: normal; 1: check, stop; 2: restart; -1: exit
	 	self.go_on = True	

	def str2list(self, s):
		return [float(x) for x in s.split('[')[1].split(']')[0].split(',')]
			
	def sendmsg(self, s):
		''' send long message, protocol 
			# -> len
			# <- start
			# -> msg
			# <- ok
		'''
		length = len(s)
		self.skt_client.sendall(str(length))
		self.skt_client.recv(self.buffsize)
		self.skt_client.sendall(s)
		self.skt_client.recv(self.buffsize)

	def recvmsg(self):
		''' recv long message, protocol 
			# <- len
			# -> start
			# <- msg
			# -> ok
		'''
		s = ''
		length = int(self.skt_client.recv(self.buffsize))
		self.skt_client.sendall('Start')
		while(length > 0):
			msg = self.skt_client.recv(self.buffsize)
			s += msg
			length -= len(msg)
		self.skt_client.sendall('OK')
		# print s
		return s

	def log(self, s):
		global START_TIME
		print('%f: %s' % (time.time() - START_TIME, s))

	# listener starts, listen to the worker
	def run(self):
		self.skt = socket(AF_INET,SOCK_STREAM)		
		addr = (self.host, self.port_base + self.index)		
		self.skt.bind(addr)
		self.skt.listen(3) # the max allowed tcp requre number.	
		while True:
			self.listen()
		self.skt.close()

	# main function for each listener,
	def listen(self):
		self.log('Wait for %dth connection ... ' % self.index)
		self.skt_client, addr_client = self.skt.accept()
		self.log('Connection from :' + str(addr_client))
		self.log('Listener  %d starts to work' % self.index)

		# send the global time
		global START_TIME
		self.skt_client.sendall('%f, %f' % (START_TIME, time.time())) # send current time to align
		self.skt_client.recv(self.buffsize)

		# if this is the first worker, need to initailize parameters
		global isfirst, parameters
		mutex.acquire()
		if(isfirst):
			isfirst = False
			self.skt_client.sendall('OK')
			while True:
				msg = self.recvmsg()
				if('$' == msg[0]):
					break
				tmp_l = np.copy(np.frombuffer(msg, dtype=np.float32))
				parameters.append(tmp_l)	
				v_para.append(np.zeros_like(tmp_l))
				# debug 
				self.log('Communication size: %fB' % (sys.getsizeof(msg)))
				self.log('Storage size: %fB' % (sys.getsizeof(tmp_l)))
			# end while
		else:
			self.skt_client.sendall('No')
			self.skt_client.recv(self.buffsize)
			for i in xrange(len(parameters)):
				self.sendmsg(parameters[i].tobytes())
			self.skt_client.sendall('$')
		mutex.release()

		# compute the size of model
		totalsize = 0.0
		for x in parameters:
			totalsize += sys.getsizeof(x)
		self.log("The size of model: %fMB" % (totalsize / (1024.0 * 1024.0)))

		# main loop: start to listen to the port
		global commit_cnt_list
		while self.go_on:
			msg = self.skt_client.recv(self.buffsize)
			if('Commit' in msg):
				commit_cnt_list[self.index] += 1
				self.skt_client.sendall("start commit")
				self.recv_commit()
				self.return_commit()
				self.check()
			elif('Exit' in msg):
				break
			else:
				self.log ('Error:%s' % msg); return	
		# end while

	def recv_commit(self):
		global parameters, worker_num, mu, eta, MOMENTUN
		global worker_loss_list, original_worker_speed_list, worker_step_list, computation_time_list, worker_speed_list, num_of_local_update
		# recv and update global parameters
		for i in xrange(len(parameters)):
			msg = self.recvmsg()
			new_l = np.copy(np.frombuffer(msg, dtype=np.float32))
			# add a momentun term here
			if(MOMENTUN):
				v_para[i] = mu * v_para[i] + eta * new_l
				parameters[i] = parameters[i] + v_para[i]
			else:
				parameters[i] = parameters[i] + new_l / float(worker_num)
		# record trival information
		msg = self.skt_client.recv(self.buffsize).split(',')
		original_worker_speed_list[self.index] = float(msg[0])
		worker_speed_list[self.index] = float(msg[3])
		worker_loss_list[self.index] = float(msg[1])
		worker_step_list[self.index] = float(msg[2])
		computation_time_list[self.index] += float(msg[5])	
		num_of_local_update[self.index] = int(msg[4])

	def return_commit(self):
		global parameters, class_cnt
		# send overall parameters
		for i in xrange(len(parameters)):
			self.sendmsg(parameters[i].tobytes())
		#send overall min class s
		self.skt_client.sendall('endCommit')

	def check(self):
		''' 
			decide whether need to check, according to the global decision
			or stop the training process
		'''
		self.skt_client.recv(self.buffsize)
		# check --- send some message
		if (self.isCheckOrStop == 1):
			self.ps_process_check()
		elif (self.isCheckOrStop == -1):
			self.skt_client.sendall('Stop')
			self.go_on = False
		elif(self.isCheckOrStop == 3):
			self.ps_process_check(isBlock = False)
		else:
			self.skt_client.sendall('No')

	def ps_process_check(self, isBlock = True):
		# sync !!! wait all workers stop
		global checkInCnt, commit_overhead_list, blocked_time_list
		tmp_time = time.time()
		if(isBlock):
			mutex.acquire()
			checkInCnt += 1
			mutex.release()
			while(self.isCheckOrStop != 2):
				pass
		
		# restart 
		self.skt_client.sendall('Check')

		# change the role from receiver to sender
		msg = self.skt_client.recv(self.buffsize).split(',')
		self.log('The %dth listener check, msg: %s' % (self.index, msg))
		
		# send delta c target 
		self.skt_client.sendall(str(-expect_commit[self.index]))
		self.isCheckOrStop = 0 # normal processing

		# record trival information
		blocked_time = time.time() - tmp_time
		commit_overhead_list[self.index] = float(msg[0])
		blocked_time_list[self.index] += blocked_time

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

		global class_num, worker_num
		global commit_cnt_list, expect_commit, commit_overhead_list,blocked_time_list
		global worker_loss_list, original_worker_speed_list, worker_step_list, computation_time_list, worker_speed_list, num_of_local_update
		worker_num = int(_total_worker_num)
		class_num = int(_class_num)

		expect_commit = [0 for _ in xrange(worker_num)]
	
		# for record
		original_worker_speed_list = [0.0 for _ in xrange(worker_num)]
		worker_speed_list = [0.0 for _ in xrange(worker_num)]
		worker_step_list = [0 for _ in xrange(worker_num)]
		worker_loss_list = [0.0 for _ in xrange(worker_num)]
		commit_cnt_list = [0 for _ in xrange(worker_num)]
		num_of_local_update = [0 for _ in xrange(worker_num)]
		computation_time_list = [0.0 for _ in xrange(worker_num)]
		commit_overhead_list = [0.0 for _ in xrange(worker_num)]
		blocked_time_list = [0.0 for _ in xrange(worker_num)]

		global beta
		with tf.device('/cpu:0'):
			beta = tf.get_variable(name='beta', shape=[class_num], initializer=tf.ones_initializer(), dtype=tf.float32, trainable=False)

		self.check_period = float(_check_period)
		self.max_frequency = 10	#per check_period
		self.base_dir = _base_dir

		# record to files
		self.f_log = open(os.path.join(self.base_dir + 'ps_log_usp.txt'), 'w')
		self.f_orig_speed = open(os.path.join(self.base_dir + 'ps_hete_usp.txt'), 'w')
		self.f_speed = open(os.path.join(self.base_dir + 'ps_speed_usp.txt'), 'w')
		self.f_cmp_time = open(os.path.join(self.base_dir + 'ps_cmp_time_usp.txt'), 'w')
		self.f_commit_overhead = open(os.path.join(self.base_dir + 'ps_commit_overhead_usp.txt'), 'w')
		self.f_blocked_time = open(os.path.join(self.base_dir + 'ps_blocked_time_usp.txt'), 'w')
		self.f_num_local_update = open(os.path.join(self.base_dir + 'ps_num_local_update_usp.txt'), 'w')
		
		self.f_global_loss = open(os.path.join(self.base_dir + 'ps_global_loss_usp.txt'), 'w')
		self.f_global_eval = open(os.path.join(self.base_dir + 'ps_global_eval_usp.txt'), 'w')
		
		# for prediction and evaluation
		self.predict_cnt = [0 for _ in xrange(class_num)]
		self.predict_rst = [0 for _ in xrange(class_num)]
		self.global_eval_rst = [0.0 for _ in xrange(class_num + 1)]

		self.host = _host 
		self.port_base = int(_port_base)
		self.band_width_limit = _band_width_limit
		self.training_end = _training_end

		self.global_loss = None
		self.trainingEndorNot = False
		self.global_loss_q = LossQueue(maxsize=WINDOW_LENGHTH_OF_LOSS)

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

	def loadmodel(self):
		global parameters
		index = 0
		for v in tf.trainable_variables():
			shape = v.get_shape()
			load_v = np.array(parameters[index]).reshape(shape)
			v.load(load_v, self.sess)
			index += 1

	def evaluation(self, eval_time = 10):
		global class_num, worker_step_list, START_TIME
		cur_time = time.time() - START_TIME
		self.global_loss = 0.0
		for _ in xrange(eval_time):
			_, _, _, _, test_images, test_labels = self.func_fetch_data(Train = False, Eval = False, Test = True)
			predictions, _loss = self.sess.run(
				[self.top_k_op, self.loss],
				feed_dict={self.X: test_images, self.Y: test_labels})
			# print(predictions)
			self.global_loss += _loss
			for i in xrange(len(test_labels)):
				self.predict_cnt[test_labels[i]] += 1
				if(predictions[i]):
					self.predict_rst[test_labels[i]] += 1
		self.global_loss = self.global_loss / float(eval_time)
		self.log("Global loss: %f\tTotal accuracy: %20.19f" %\
			(self.global_loss, sum(self.predict_rst) / float(sum(self.predict_cnt))))
		
		## calculate accuracy for each class
		self.global_eval_rst[-1] = sum(self.predict_rst) / float(sum(self.predict_cnt))
		for i in xrange(class_num):
			if(self.predict_cnt[i] == 0):
				self.global_eval_rst[i] = -1.0
			else:
				self.global_eval_rst[i] = float(self.predict_rst[i]) / self.predict_cnt[i]
		self.f_global_eval.write('%020.10f: %s\n' % (cur_time, str(self.global_eval_rst)))
		self.f_global_loss.write('%020.10f, %020.10f, %020.10f, %030.20f\n' % 
							(cur_time, average(worker_step_list), sum(worker_step_list), self.global_loss))
		self.f_global_eval.flush()
		self.f_global_loss.flush()

		self.predict_cnt = [0 for _ in xrange(class_num)]
		self.predict_rst = [0 for _ in xrange(class_num)]

	def get_global_loss(self):
		self.loadmodel()
		self.evaluation(10)
		self.global_loss_q.put(self.global_loss)
		isEnd, var = self.global_loss_q.isConverge()
		self.log('Current variance: %f' % (var if var else 0))
		if(isEnd):
			self.trainingEndorNot = True

	def record_info(self):
		global START_TIME, commit_cnt_list
		cur_time = time.time() - START_TIME
		global original_worker_speed_list, worker_speed_list
		global worker_step_list, worker_loss_list, num_of_local_update
		global computation_time_list, commit_overhead_list, blocked_time_list
		
		self.f_orig_speed.write('%020.10f: %s\n' % (cur_time, str(original_worker_speed_list)))
		self.f_speed.write('%020.10f: %s\n' % (cur_time, str(worker_speed_list)))
		self.f_log.write('%020.10f: step = %s\tloss= %s\tcommit_cnt=%s\n' % 
			(cur_time, str(worker_step_list), str(worker_loss_list), str(commit_cnt_list)))
		self.f_num_local_update.write('%020.10f: %s\n' % (cur_time, str(num_of_local_update)))
		self.f_cmp_time.write('%020.10f: %s\n' % (cur_time, str(computation_time_list)))	
		self.f_commit_overhead.write('%020.10f: %s\n' % (cur_time, str(commit_overhead_list)))
		self.f_blocked_time.write('%020.10f: %s\n' % (cur_time, str(blocked_time_list)))
			
		self.f_orig_speed.flush()
		self.f_speed.flush()
		self.f_log.flush()
		self.f_num_local_update.flush()
		self.f_cmp_time.flush()
		self.f_commit_overhead.flush()
		self.f_blocked_time.flush()
		

	# stop all threads
	def allStop(self, ps_t):
		global worker_num, checkInCnt
		for i in xrange(worker_num):
			ps_t[i].isCheckOrStop = 1
		while(checkInCnt != worker_num):
			pass
		checkInCnt = 0

	# start all threads
	def allStart(self, ps_t):
		global worker_num
		for i in xrange(worker_num):
			ps_t[i].isCheckOrStop = 2

	def onlineSearch1Min(self, ps_t, c_target, last_loss, last_loss_time):
		'''
			search with one configuration for 1 min
		'''
		global START_TIME, expect_commit
		expect_commit = [c_target for _ in expect_commit]
		cnt = 0

		# start to run the configuration
		self.allStart(ps_t)
		while(cnt <= self.check_period / 2):
			cnt += 1
			time.sleep(1)
		self.get_global_loss()
		last_loss.append(self.global_loss)
		last_loss_time.append(time.time() - START_TIME)
		while(cnt <= self.check_period):
			cnt += 1
			time.sleep(1)
		self.allStop(ps_t)
		# end to run this configuration

		self.get_global_loss()
		last_loss.append(self.global_loss)
		last_loss_time.append(time.time() - START_TIME)
		reward = self.get_reward(last_loss, last_loss_time)
		self.log("Delta C_target: %f\treward: %f (last_loss list: %s\t time list: %s" % (c_target, reward, str(last_loss), str(last_loss_time)))
		print
		last_loss = [last_loss[-1]]
		last_loss_time = [last_loss_time[-1]]
		return reward, last_loss, last_loss_time
				
	def get_reward(self, last_loss, last_loss_time):
		def func(x, a, b, c):
			return 1.0 / (a * a * x + b) + c
		popt, pcov = curve_fit(func, last_loss_time, last_loss, maxfev=5000000)
		a = popt[0]
		b = popt[1]
		c = popt[2]
		if(self.training_end == c):
			return - 100000.0
		tmp = (1.0 / (self.training_end - c) - b)
		if(tmp == 0):
			return - 100001.0
		else:
			return a * a / tmp

	# main run of the ps scheduler
	def run(self):
		global worker_num, class_num, class_cnt
		global expect_commit
		
		# create listeners and launch them
		ps_t = [PSThread(i, self.host, self.port_base) for i in xrange(worker_num)]
		for i in xrange(worker_num):
			ps_t[i].start()

		# huhanpeng: delete ???
		# the commit cnt should be limited in case of exceeding the limit of the bandwidth
		COST_PER_COMMIT = 78.0 # COST_PER_COMMIT is 78 M
		if(self.band_width_limit == None):
	 		c_target_max_by_bandwidth = 100000.0
		else:
			# specify the constrait of bandwidth in the form of x M/s
			c_target_max_by_bandwidth = float(sys.argv[1]) * self.check_period / COST_PER_COMMIT
		
		global isfirst
		check_cnt = 0 # counter for check_point
		epochCnt = 0 # the counter for epoch
		forceSearch = True # used to avoid divergence
		gridSearchTime = 0.0
		lastSearchCnt = check_cnt
			
		# huhanpeng: delete ?? test
		# self.check_period = 120
		c_target = 1
		expect_commit = [c_target for _ in expect_commit]
		
		while(not self.trainingEndorNot):
			time.sleep(1)
			if(not isfirst):
				# isfirst is to ensure running after the first worker connects to the PS
				check_cnt = check_cnt + 1
				if(forceSearch or (check_cnt - lastSearchCnt >= CHECKPOINT_PER_EPOCH)):
					global START_TIME
					tmp_time = time.time()
					epochCnt += 1

					print
					self.log('######################### Search ##################################')
					self.log("The %dth search\tcheckpoint: %d" % (epochCnt, check_cnt))	
					self.allStop(ps_t)
					self.get_global_loss()
					last_loss = [self.global_loss]	
					last_loss_time = [time.time() - START_TIME]
					cur_reward, last_loss, last_loss_time = self.onlineSearch1Min(ps_t, c_target, last_loss, last_loss_time)
					if(not self.trainingEndorNot): 
						next_reward, last_loss, last_loss_time = self.onlineSearch1Min(ps_t, c_target+1, last_loss, last_loss_time)
					else:
						self.log('end in search')

					# if next_reward c_target+1 is better, continue to search c_target + 2
					while((not self.trainingEndorNot) and next_reward > cur_reward):
						cur_reward = next_reward
						c_target += 1 
						next_reward, last_loss, last_loss_time = self.onlineSearch1Min(ps_t, c_target+1, last_loss, last_loss_time)
					# now, current c_target is better
					expect_commit = [c_target for _ in expect_commit]

					self.allStart(ps_t)
					forceSearch = False
					lastSearchCnt = check_cnt
					gridSearchTime += time.time() - tmp_time
					self.log("Accu Search Time: %f\nOpt c_target: %f"%(gridSearchTime, c_target))
					self.log("USP:expect_commit %s\n" % (str(expect_commit)))

					self.record_info()
					self.log('#################################################################')
					print
				# end if # epoch

				if(check_cnt % self.check_period == 0):
					## for record
					self.record_info()
					self.log("The %dth check_period; expect_commit: %s" % (check_cnt / self.check_period, str(expect_commit)))
					
					# evaluation
					last_global_loss = self.global_loss
					self.get_global_loss()
					if(last_global_loss != None and (self.global_loss - last_global_loss) / last_global_loss >= 0.3):
						forceSearch = True
					for i in xrange(worker_num):
						ps_t[i].isCheckOrStop = 3	
				#end if
			#end if first
		# end while
		self.log("PS ends")
		for i in xrange(worker_num):
			ps_t[i].isCheckOrStop = -1
		for i in xrange(worker_num):
			ps_t[i].join()
		self.f_orig_speed.close()
		self.f_speed.close()
		self.f_log.close()
		self.f_num_local_update.close()
		self.f_cmp_time.close()
		self.f_commit_overhead.close()
		self.f_blocked_time.close()
		self.f_global_loss.close()
		self.f_global_eval.close()




