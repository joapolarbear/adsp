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


def sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None):
	# beta list, whose components are the cost of corresponding class
	global beta
	# call tensorflow api to calculate the original loss
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
					_sentinel,
					labels,
					logits,
					name
	)
	# calculate the coefficient for each example in the mini-batch
	coefficient = tf.gather(params=beta, indices=labels)
	# return the weighted loss
	return tf.multiply(cross_entropy, coefficient)

def softmax_cross_entropy_with_logits(self, _sentinel=None, labels=None, logits=None, dim=-1, name=None):
	global beta
	cross_entropy = self.softmax_cross_entropy_with_logits(
						_sentinel,
						labels,
						logits,
						dim,
						name
					)
	coefficient = tf.gather(params=beta, indices=labels)
	return tf.multiply(cross_entropy, coefficient)

def softmax_cross_entropy_with_logits_v2(_sentinel=None, labels=None, logits=None, dim=-1, name=None):
	global beta
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
					_sentinel,
					labels,
					logits,
					dim,
					name	
				)
	coefficient = tf.gather(params=beta, indices=labels)
	return tf.multiply(cross_entropy, coefficient)

def sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None):
	global beta
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
					_sentinel,
					labels,
					logits,
					name
	)
	coefficient = tf.gather(params=beta, indices=labels)
	return tf.multiply(cross_entropy, coefficient)

def weighted_cross_entropy_with_logits(targets, logits, pos_weight, name=None):
	global beta
	cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
		targets,
		logits,
		pos_weight,
		name=None
	)
	coefficient = tf.gather(params=beta, indices=labels)
	return tf.multiply(cross_entropy, coefficient)

def sigmoid_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None):
	global beta
	cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
		_sentinel,
		labels,
		logits,
		name
	)
	coefficient = tf.gather(params=beta, indices=labels)
	return tf.multiply(cross_entropy, coefficient)

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
		_port_base = 14200):

		self.No = int(_worker_index)
		# self.check_period = float(_check_period)
		self.check_period = 120
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

		self.min_j = 0
		# self.beta = [1 for _ in xrange(self.class_num)]
		self.commit_cnt = 0 # record the total commit number
		self.class_cnt = [0 for _ in xrange(self.class_num)]

		## for prediction
		self.predict_cnt = [0 for _ in xrange(self.class_num)]
		self.predict_rst = [0 for _ in xrange(self.class_num)]
		self.eval_rst = [0.0 for _ in xrange(self.class_num + 1)] # last elem is the overall accuracy

		self.f_log = open(os.path.join(self.base_dir + 'wk_%d_usp.txt' % (self.No)), 'w')
		self.f_pre = open(os.path.join(self.base_dir + 'wk_%d_usp_pred.txt' % (self.No)), 'w')

		self.parameter = []  # a list of parameters, parameters are np.array
		self.para_shape = []


	def connect2PS(self):
		# connect to the PS
		addr = (self.host, self.port_base + self.No)
		self.skt = socket(AF_INET,SOCK_STREAM)
		self.skt.connect(addr)
		self.buffsize = 1024
		self.start_time = float(self.skt.recv(self.buffsize))
		self.skt.sendall("Recv start_t")

		msg = self.skt.recv(self.buffsize)
		if('OK' in msg):
			for v in tf.trainable_variables():
				self.parameter.append(v.eval(session = self.sess))
				shape = v.get_shape()
				self.para_shape.append(shape)
				self.sendmsg(str(self.parameter[-1].flatten().tolist()))
			self.sendmsg('$')
		else:
			self.skt.sendall('Load')
			# Receive the overall parameters from the PS
			for v in tf.trainable_variables():
				l = self.str2list(self.recvmsg())
				shape = v.get_shape()
				print("shape: %s\tsize of elements: %f" % (str(shape), sys.getsizeof(v[0])))
				self.para_shape.append(shape)
				self.parameter.append(np.array(l).reshape(shape))
			self.skt.recv(self.buffsize)
			# update the local mode
			index = 0
			for v in tf.trainable_variables():
				v.load(self.parameter[index], self.sess)
				index += 1
		print("Worker %d successfully connct to the "
		 	"parameter server [%s:%d]" % (self.No, self.host, (self.port_base + self.No)))

	''' label = -1. clear class_cnt
		or update corresponding class_cnt
	'''
	def update_class(self, _labels = None):
		if(_labels == None):
			self.class_cnt = [0 for _ in xrange(self.class_num)]
		else:
			for i in xrange(len(_labels)):
				self.class_cnt[_labels[i]] += 1
		# end if

	def str2list(self, s):
		return [float(x) for x in s.split('[')[1].split(']')[0].split(',')]

	''' send long message, protocol 
		# -> len
		# <- start
		# -> msg
		# <- ok
	'''
	def sendmsg(self, s):
		length = len(s)
		self.skt.sendall(str(length))
		self.skt.recv(self.buffsize)
		self.skt.sendall(s)
		self.skt.recv(self.buffsize)

	''' recv long message, protocol 
		# <- len
		# -> start
		# <- msg
		# -> ok
	'''
	def recvmsg(self):
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
		self.train_writer = tf.summary.FileWriter(self.base_dir + '/board/train', self.sess.graph)
		self.eval_writer = tf.summary.FileWriter(self.base_dir + '/board/eval', self.sess.graph)

		self.merged =  _merged
		self.train_op = _train_op
		self.loss = _loss
		self.global_step = _global_step
		self.X = _feed_X
		self.Y = _feed_Y
		self.feed_prediction = _feed_prediction
		self.top_k_op = _top_k_op

		# self.beta = _beta

	def run(self, simulate = 0):
		self.connect2PS()
		ep = 0
		time_after_a_commit = time.time()

		# for avoiding divergence
		forceCommit = False
		lastLoss = None
		divergeThreshold = 5
		# while not self.sess.should_stop():
		while (ep < self.max_steps):
			time_step_start = time.time()
			ep += 1
			# if(simulate):
			time.sleep(simulate)
			train_X, train_Y, eval_X, eval_Y, test_X, test_Y = self.func_fetch_data(Train = True, Eval = True, Test = True)
			cur_train_labels_list = train_Y.tolist()
			# train one mini batch
			tmp_time = time.time()
			train_summary, _, cur_loss, cur_step = \
				self.sess.run([
					self.merged, 
					self.train_op, 
					self.loss,  
					self.global_step], 
				feed_dict = {
					self.X: train_X, 
					self.Y: train_Y})
			hete_time = time.time() - tmp_time
			

			self.update_class(cur_train_labels_list)	# update class_cnt
			cur_time = time.time() - self.start_time
			self.train_writer.add_summary(train_summary, int(cur_time))	
			self.f_log.write('%020.10f, %-10d, %030.20f\n' % (cur_time, ep, cur_loss))
			self.f_log.flush()

			# evaluation
			if(ep % 10 == 0):
				eval_summary, _, eval_loss = \
					self.sess.run([
						self.merged, 
						self.train_op, 
						self.loss], 
					feed_dict = {
						self.X: eval_X, 
						self.Y: eval_Y})
				cur_time = time.time() - self.start_time
				self.eval_writer.add_summary(eval_summary, int(cur_time))	
				print('USP_WK{} - Step:{}\tTime:{}\nbase_time_step:{}s\n\tTrain loss: {}\tEval loss: {}'\
					.format(
						self.No, 
						ep, 
						cur_time, 
						self.base_time_step, 
						cur_loss, 
						eval_loss))
			else:
				print('USP_WK{} - Step:{}\tTime:{}\nbase_time_step:{}s\tTrain loss: {}'\
					.format(
						self.No, 
						ep, 
						cur_time, 
						self.base_time_step, 
						cur_loss))

			# record the a time	
			time_one_step = time.time() - time_step_start 			
			time_span = time.time() - time_after_a_commit
			print
			print("1:Time per batch: %f" % (hete_time))
			print("time one step: %f" % (time_one_step))
			print("time from last commit: %f" % (time_span))
			print

			#  commit to the PS 
			if(time_span > self.base_time_step):
				self.wk_process_commit(cur_time, cur_loss, hete_time, time_one_step, ep, time_span)
				# whether to check ? 
				self.skt.sendall('Check?')
				msg = self.skt.recv(self.buffsize)
				print(msg)
				if('Check' in msg):
					self.wk_process_check(cur_time, ep, test_X, test_Y)
				elif('Stop' in msg):
					break
				# recalculate time_after_a_commit after one commit
				time_after_a_commit = time.time()
			# end commit
		#end while
		print('USP_WK{} stops training - Step:{}\tTime:{}'.format(self.No, ep, cur_time))
		self.evaluation(cur_time, 100)

	def wk_process_commit(self, cur_time, cur_loss, hete_time, time_one_step, cur_step, time_span):
		self.commit_cnt += 1
		self.skt.sendall('Commit')
		self.skt.recv(self.buffsize)

		tmp_time = time.time()
		# Sum and send all local updates
		index = 0
		communication_size = 0
		for v in tf.trainable_variables():
			cur_parameter = v.eval(session = self.sess)
			# Send the result multiplied with gamma to the PS		
			# self.sendmsg(str((self.gamma * ((cur_parameter - self.parameter[index]).flatten())).tolist()))
			l = str((((cur_parameter - self.parameter[index]).flatten())).tolist())
			communication_size += sys.getsizeof(l)
			self.sendmsg(l)
			# self.skt.sendall(str((((cur_parameter - self.parameter[index]).flatten())).tolist()))
			self.parameter[index] = cur_parameter[:]
			index += 1
		# end for
		print("  -- The size of transmitted Parameters: %d M" % (float(communication_size)/(1024.0 * 1024.0)))
		print("2:Time of extracting and sending parameters: %f" % (time.time() - tmp_time))

		tmp_time = time.time()
		## send some trival message
		l = "%20.15f, %20.15f, %d, %20.15f, %20.15f" % (hete_time, cur_loss, cur_step, time_one_step, time_span)
		print("  -- The size of (send some trival message): %d M" % (float(sys.getsizeof(l))/(1024.0 * 1024.0)))
		self.skt.sendall(l)
		print("3:Time of sending trival messages: %f" % (time.time() - tmp_time))

		tmp_time = time.time()
		##  Receive the overall parameters from the PS
		# communication_size = 0
		for i in xrange(len(self.parameter)):
			l = self.recvmsg()
			# communication_size += sys.getsizeof(l)
			self.parameter[i] = np.array(self.str2list(l)).reshape(self.para_shape[i])
		# print("  -- The size of transmitted Parameters: %d M" % (float(communication_size)/(1024.0 * 1024.0)))
		# receive the min class index
		self.min_j = int(self.skt.recv(self.buffsize))
		print("4:Time of recv parameters: %f" % (time.time() - tmp_time))

		tmp_time = time.time()
		# Update the local mode
		# self.saver.save(self.sess, "./tmp/check.ckpt")
		index = 0
		for v in tf.trainable_variables():
			# assign_op = v.assign(self.parameter[index])
			tt = time.time()
			v.load(self.parameter[index], self.sess)
			# self.sess.run(assign_op)
			print("		-- time: %f\tsize: %f" %(time.time() - tt, sys.getsizeof(self.parameter[index])))
			# print('$$$$$$$$$$$$$$$')
			# print (v.eval(session = self.sess))
			# print self.parameter[index]
			index += 1
		print("5:Time of load parameters: %f" % (time.time() - tmp_time))
		print("***************\nUSP_WK{}:Commit {} finished\n***********".format(self.No, self.commit_cnt))

	def wk_process_check(self, cur_time, cur_step, test_X, test_Y):
		## Evaluation
		self.evaluation(cur_time)

		## send msg as below
		# Send the class counters and average loss cnt to the PS, and the iteration_time
		self.sendmsg(str(self.class_cnt))
		print("class_cnt %d: %s" % (self.No, self.class_cnt))
		self.sendmsg(str(self.eval_rst))

		# change the role from sender to receiver
		self.skt.send("start receive msg")

		# Receive new beta, gamma, from the PS
		_cur_beta = self.str2list(self.recvmsg())	
		# update local beta tensor
		cur_beta = [1 + ((x - 1) / math.sqrt(1 + (cur_step / 500))) for x in _cur_beta]
		# cur_beta = [1 for x in _cur_beta]
		global beta
		beta = cur_beta
		
		# receive target commit No 
		expect_cNo = float(self.skt.recv(self.buffsize))
		print("expect_cNo: %f \n beta:%s" % (expect_cNo, str(beta)))
		
		# # update the step size D
		# # self.D = min(30, max(8, self.adjust_step(expect_cNo, self.D, self.commit_cnt, cur_step)))
		# self.D = max(1.0, self.adjust_step(expect_cNo, self.D, self.commit_cnt, time.time() - last_check_time))

		# update the base_time_step
		self.base_time_step = self.check_period / expect_cNo

		
		
		# whether to clear these counter ??? : huhanpeng
		self.update_class()

	def evaluation(self, cur_time = None, eval_time = 10):
		for _ in xrange(eval_time):
			# _eval_images, _eval_labels = self.sess.run([self.eval_images, self.eval_labels])
			# _eval_images, _eval_labels = self.fetch_data()
			# fetch test (X, Y) for next evaluation
			_, _, _, _, test_X, test_Y = self.func_fetch_data(Train = False, Eval = False, Test = True)

			if(self.feed_prediction is not None):
				predictions = self.sess.run(
				self.top_k_op, 
				feed_dict={
					self.X: test_X, 
					self.Y: test_Y,
					self.feed_prediction: test_X})
			else:
				predictions = self.sess.run(
					self.top_k_op, 
					feed_dict={
						self.X: test_X, 
						self.Y: test_Y})
			# print(predictions)
			for i in xrange(len(test_Y)):
				self.predict_cnt[test_Y[i]] += 1
				if(predictions[i]):
					self.predict_rst[test_Y[i]] += 1
			
		print("Time: %20.10f \tTotal accuracy: %20.19f\n" %\
			(cur_time, sum(self.predict_rst) / float(sum(self.predict_cnt))))
		
		self.eval_rst[-1] = sum(self.predict_rst) / float(sum(self.predict_cnt))
		self.f_pre.write("Time: %20.10f \tTotal accuracy: %20.19f\n" % (cur_time, self.eval_rst[-1]))			
		for i in xrange(self.class_num):
			if(self.predict_cnt[i] == 0):
				self.f_pre.write("%20.19f " % (-1.0))
				self.eval_rst[i] = -1.0
				# print("%20.19f " % (-1.0))
			else:
				self.eval_rst[i] = float(self.predict_rst[i]) / self.predict_cnt[i]
				self.f_pre.write("%20.19f " % (self.eval_rst[i]))
				# print("%20.19f " % (float(self.predict_rst[i]) / self.predict_cnt[i]))	
		self.f_pre.write('\n')
		self.predict_cnt = [0 for _ in xrange(self.class_num)]
		self.predict_rst = [0 for _ in xrange(self.class_num)]

# global variables for parameter server
worker_num = 3
class_num = 10

delta_commit_cnt = []
# gamma_cnt = [0 for _ in xrange(worker_num)]
expect_commit = []
loss_for_each_worker = []
hete_for_each_worker = []
step_for_each_worker = []
comp_time_for_each_worker = []
speed_for_each_worker = []

eval_rst = []
class_cnt = []

parameters = []
v_para = []
isfirst = True
start_time = time.time()

mu = 0.6
eta = None
checkInCnt = 0
# _time_list = []
# _loss_list = []
# _tmp_time_list = []
# _tmp_loss_list = []


# f_loss = open(base_dir + 'loss_ps_usp.txt', 'w')

# max_c = 100000
mutex = threading.Lock()

def average(seq): 
	return float(sum(seq)) / len(seq)
	

	# end sess
# end class
class PSThread(threading.Thread):	
	def __init__(self, worker_index, _host, _port_base):
		threading.Thread.__init__(self)
		self.host = _host
	 	self.port_base = _port_base
	 	self.buffsize = 1024
	 	self.index = worker_index
	 	self.isCheckOrStop = 0	
	 	self.go_on = True	


	def str2list(self, s):
		return [float(x) for x in s.split('[')[1].split(']')[0].split(',')]
				
	''' send long message, protocol 
		# -> len
		# <- start
		# -> msg
		# <- ok
	'''
	def sendmsg(self, s):
		length = len(s)
		self.skt_client.sendall(str(length))
		self.skt_client.recv(self.buffsize)
		self.skt_client.sendall(s)
		self.skt_client.recv(self.buffsize)

	''' recv long message, protocol 
		# <- len
		# -> start
		# <- msg
		# -> ok
	'''
	def recvmsg(self):
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

	''' Thread start '''
	def run(self):
		self.skt = socket(AF_INET,SOCK_STREAM)		
		addr = (self.host, self.port_base + self.index)		
		self.skt.bind(addr)
		self.skt.listen(3) # the max allowed tcp requre number.	
		while True:
			self.listen()
		self.skt.close()

	def listen(self):
		print('Wait for %dth connection ... ' % self.index)
		self.skt_client, addr_client = self.skt.accept()
		print('connection from :' + str(addr_client))
		print('listener start to work %d' % self.index)

		# check_t = threading.Thread(target = self.check, args = ())
		# check_t.start()
		global start_time
		self.skt_client.sendall(str(start_time))
		self.skt_client.recv(self.buffsize)

		# if this is first worker, need to initailize parameters
		global isfirst, parameters
		mutex.acquire()
		if(isfirst):
			isfirst = False
			self.skt_client.sendall('OK')
			while True:
				msg = self.recvmsg()
				if('$' in msg):
					break
				tmp_l = self.str2list(msg)
				parameters.append(tmp_l)	
				v_para.append([0.0 for _ in tmp_l])
			# end while
		else:
			self.skt_client.sendall('No')
			self.skt_client.recv(self.buffsize)
			for i in xrange(len(parameters)):
				self.sendmsg(str(parameters[i]))
			self.skt_client.sendall('$')
		mutex.release()

		totalsize = 0.0
		for x in parameters:
			# print len(x)
			totalsize += sys.getsizeof(x)
		print("The size of model: %fMB" % (totalsize / (1024.0 * 1024.0)))

		# start to listen to the port
		global delta_commit_cnt
		while self.go_on:
			msg = self.skt_client.recv(self.buffsize)
			if('Commit' in msg):
				delta_commit_cnt[self.index] += 1
				self.skt_client.sendall("start commit")
				self.recv_commit()
				self.return_commit()
				self.check()
			elif('Exit' in msg):
				break
			else:
				print ('error:%s' % msg); return

		# convert str to 1 dimention list!!!!
		# s.split('[')[1].split(']')[0].split(',')
		time.sleep(1)	
		# end while

	def recv_commit(self):
		global parameters, worker_num, mu, eta
		global loss_for_each_worker, hete_for_each_worker, step_for_each_worker, comp_time_for_each_worker, speed_for_each_worker
		# before update the 
		# recv and update parameters
		for i in xrange(len(parameters)):
			new_l = self.str2list(self.recvmsg())
			# add a momentun term here
			parameters[i] = [parameters[i][x] + new_l[x] / float(worker_num) for x in xrange(len(new_l))]
			# parameters[i] = [parameters[i][x] + new_l[x] for x in xrange(len(new_l))]
		msg = self.skt_client.recv(self.buffsize).split(',')

		hete_for_each_worker[self.index] = float(msg[0])
		loss_for_each_worker[self.index] = float(msg[1])
		step_for_each_worker[self.index] = float(msg[2])
		comp_time_for_each_worker[self.index] += float(msg[4])
		speed_for_each_worker[self.index] = float(msg[3])


	def return_commit(self):
		global parameters, class_cnt
		# send overall parameters
		for i in xrange(len(parameters)):
			self.sendmsg(str(parameters[i]))
		#send overall min class index
		self.skt_client.sendall(str(class_cnt.index(min(class_cnt))))

	def check(self):
		self.skt_client.recv(self.buffsize)
		# check --- send some message
		if (self.isCheckOrStop == 1):
			self.ps_process_check()
		elif (self.isCheckOrStop == -1):
			self.skt_client.sendall('Stop')
			self.go_on = False
		else:
			self.skt_client.sendall('No')

	def ps_process_check(self):
		global class_cnt, eval_rst

		# sync !!!
		global worker_num, checkInCnt
		mutex.acquire()
		checkInCnt += 1
		mutex.release()
		while(self.isCheckOrStop != 2):
			pass
		# print("%d go" % self.index)

		self.skt_client.sendall('Check')
		# recv and upsta weight class cnt
		new_l = self.str2list(self.recvmsg())
		class_cnt = [class_cnt[x] + new_l[x] for x in xrange(len(new_l))]
		# recv and update loss cnt
		new_l = self.str2list(self.recvmsg())
		for x in xrange(len(eval_rst[self.index])):
			eval_rst[self.index][x] = new_l[x]

		# change the role from receiver to sender
		self.skt_client.recv(self.buffsize)

		# calculate and send beta
		K = 2.0
		# dao_class = [1.0 / float(x+1) for x in class_cnt] # x+1??: huhanpeng
		# dao_class = [x for x in class_cnt] # x+1??: huhanpeng
		# sum_dao_class = sum(dao_class)
		# beta = [1 + (K - 1) * float(x) / float(sum_dao_class) for x in dao_class]
		
		max_class_num = max(class_cnt)
		ps_beta = [float(max_class_num)/max(float(x), 1) for x in class_cnt]
		# print('Beta:', beta)
		self.sendmsg(str(ps_beta))
		# calculate and send gamma
		# self.sendmsg(str(gamma_cnt[self.index] / sum(gamma_cnt)))

		# send expect commit number in before next check
		self.skt_client.sendall(str(expect_commit[self.index]))
		self.isCheckOrStop = 0

# end class thread 
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
		_epsilon = 0.3
		):

		global class_num, worker_num
		global delta_commit_cnt, expect_commit
		global loss_for_each_worker, hete_for_each_worker, step_for_each_worker, comp_time_for_each_worker, speed_for_each_worker
		global eval_rst, class_cnt
		worker_num = int(_total_worker_num)
		global eta
		eta = 1.0 / float(worker_num)
		class_num = int(_class_num)

		delta_commit_cnt = [0 for _ in xrange(worker_num)]
		# gamma_cnt = [0 for _ in xrange(worker_num)]
		expect_commit = [0 for _ in xrange(worker_num)]
		loss_for_each_worker = [0.0 for _ in xrange(worker_num)]
		hete_for_each_worker = [0.0 for _ in xrange(worker_num)]
		step_for_each_worker = [0 for _ in xrange(worker_num)]
		comp_time_for_each_worker = [0.0 for _ in xrange(worker_num)]
		speed_for_each_worker = [0.0 for _ in xrange(worker_num)]

		
		class_cnt = [0 for _ in xrange(class_num)]
		eval_rst = [[0.0 for _ in xrange(class_num + 1)] for _ in xrange(worker_num)]

		global beta
		with tf.device('/cpu:0'):
			beta = tf.get_variable(name='beta', shape=[class_num], initializer=tf.ones_initializer(), dtype=tf.float32, trainable=False)

		self.check_period = float(_check_period)
		self.max_frequency = 10	#per check_period
		self.base_dir = _base_dir

		self.f_hete = open(os.path.join(self.base_dir + 'ps_hete_usp.txt'), 'w')
		self.f_eval = open(os.path.join(self.base_dir + 'ps_eval_usp.txt'), 'w')
		self.f_cmp_time = open(os.path.join(self.base_dir + 'ps_cmp_time_usp.txt'), 'w')
		self.f_speed = open(os.path.join(self.base_dir + 'ps_speed_usp.txt'), 'w')
		self.f_bandit_cnt = open(os.path.join(self.base_dir + 'ps_bandit_cnt_usp.txt'), 'w')

		self.f_global_loss = open(os.path.join(self.base_dir + 'ps_global_loss_usp.txt'), 'w')
		self.f_global_eval = open(os.path.join(self.base_dir + 'ps_global_eval_usp.txt'), 'w')
		self.f_commit = open(os.path.join(self.base_dir + 'ps_commit_usp.txt'), 'w')

		self.predict_cnt = [0 for _ in xrange(class_num)]
		self.predict_rst = [0 for _ in xrange(class_num)]
		self.global_eval_rst = [0.0 for _ in xrange(class_num + 1)]

		self.commit_cnt = [0 for _ in xrange(worker_num)]
		self.host = _host 
		self.port_base = int(_port_base)
		self.band_width_limit = _band_width_limit
		self.training_end = _training_end

		self.epsilon = _epsilon

		self.global_loss = None
		self.training_end_cnt = 0
		# eta = 1.0 / float(worker_num)
		# eta = 0.1

	def default_func():
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
		self.train_writer = tf.summary.FileWriter(self.base_dir + '/board/train', self.sess.graph)
		self.eval_writer = tf.summary.FileWriter(self.base_dir + '/board/eval', self.sess.graph)

		self.merged =  _merged
		self.train_op = _train_op
		self.loss = _loss
		self.global_step = _global_step
		self.X = _feed_X
		self.Y = _feed_Y
		self.feed_prediction = _feed_prediction
		self.top_k_op = _top_k_op

	def loadmodel(self, _parameters):
		index = 0
		for v in tf.trainable_variables():
			shape = v.get_shape()
			load_v = np.array(_parameters[index]).reshape(shape)
			v.load(load_v, self.sess)
			index += 1

	def evaluation(self, eval_time = 10):
		global class_num, step_for_each_worker, start_time
		cur_time = time.time() - start_time
		self.global_loss = 0.0
		for _ in xrange(eval_time):
			_, _, _, _, test_images, test_labels = self.func_fetch_data(Train = False, Eval = False, Test = True)
			# test_images, test_labels = self.sess.run([self.test_images, self.test_labels])
			# _eval_images, _eval_labels = self.sess.run([self.eval_images, self.eval_labels])
			# _eval_images, _eval_labels = self.fetch_data()
			predictions, _loss = self.sess.run(
				[self.top_k_op, self.loss],
				feed_dict={
					self.X: test_images, 
					self.Y: test_labels})
			# print(predictions)
			self.global_loss += _loss
			for i in xrange(len(test_labels)):
				self.predict_cnt[test_labels[i]] += 1
				if(predictions[i]):
					self.predict_rst[test_labels[i]] += 1
		self.global_loss = self.global_loss / float(eval_time)
		print("Time: %20.10f \tGlobal loss: %f\tTotal accuracy: %20.19f" %\
			(cur_time, self.global_loss, sum(self.predict_rst) / float(sum(self.predict_cnt))))
		
		## calculate accuracy for each class
		self.global_eval_rst[-1] = sum(self.predict_rst) / float(sum(self.predict_cnt))
		for i in xrange(class_num):
			if(self.predict_cnt[i] == 0):
				self.global_eval_rst[i] = -1.0
			else:
				self.global_eval_rst[i] = float(self.predict_rst[i]) / self.predict_cnt[i]
		self.f_global_eval.write('%s\n' % str(self.global_eval_rst))
		self.f_global_loss.write('%020.10f, %020.10f, %020.10f, %030.20f\n' % 
							(cur_time, average(step_for_each_worker), sum(step_for_each_worker), self.global_loss))
		self.f_global_eval.flush()
		self.f_global_loss.flush()

		self.predict_cnt = [0 for _ in xrange(class_num)]
		self.predict_rst = [0 for _ in xrange(class_num)]

	def record_info(self, last_avg_loss=None, epsilon_g=None):
		global start_time
		cur_time = time.time() - start_time
		global hete_for_each_worker, step_for_each_worker, eval_rst, comp_time_for_each_worker, speed_for_each_worker
		self.f_hete.write('%020.10f: %s\n' % (cur_time, str(hete_for_each_worker)))
		self.f_eval.write('%020.10f: %s\n' % (cur_time, str(eval_rst)))
		self.f_cmp_time.write('%020.10f: %s\n' % (cur_time, str(comp_time_for_each_worker)))
		self.f_speed.write('%020.10f: %s\n' % (cur_time, str(speed_for_each_worker)))
		self.f_commit.write('%020.10f: %s\n' % (cur_time, str(self.commit_cnt)))
		self.f_hete.flush()
		self.f_eval.flush()
		self.f_cmp_time.flush()
		self.f_speed.flush()
		self.f_commit.flush()

		# self.f_bandit_cnt.write('%020.10f: %s\n' % (cur_time, str(epsilon_g.counts)))
		self.f_bandit_cnt.flush()	

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

	''' online method'''
	def onlineSearch1Min(self, ps_t, c_target, last_loss, last_loss_time):
		global parameters, start_time, expect_commit
		expect_commit = [c_target for _ in expect_commit]
		cnt = 0
		self.allStart(ps_t)
		while(cnt <= self.check_period / 2):
			cnt += 1
			time.sleep(1)
		self.loadmodel(parameters)
		self.evaluation(10)
		last_loss.append(self.global_loss)
		last_loss_time.append(time.time() - start_time)
		while(cnt <= self.check_period):
			cnt += 1
			time.sleep(1)
		self.allStop(ps_t)

		self.loadmodel(parameters)
		self.evaluation(10)
		if(self.global_loss < self.training_end):
			self.training_end_cnt += 1
		last_loss.append(self.global_loss)
		last_loss_time.append(time.time() - start_time)
		reward = self.get_reward(last_loss, last_loss_time)
		print("Delta C_target: %f\treward: %f (last_loss: %s\t time: %s" % (c_target, reward, str(last_loss), str(last_loss_time)))
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


	# main run of the ps schedualer
	def run(self):
		global worker_num, class_num, class_cnt
		global expect_commit, delta_commit_cnt
		
		ps_t = [PSThread(i, self.host, self.port_base) for i in xrange(worker_num)]
		for i in xrange(worker_num):
			ps_t[i].start()

		# the commit cnt should be limited in case of exceeding the limit of the bandwidth
		COST_PER_COMMIT = 78.0 # COST_PER_COMMIT is 78 M
		if(self.band_width_limit == None):
	 		c_target_max_by_bandwidth = 100000.0
		else:
			# specify the constrait of bandwidth in the form of x M/s
			c_target_max_by_bandwidth = float(sys.argv[1]) * self.check_period / COST_PER_COMMIT
		
		global isfirst
		check_cnt = 0
		epoch = 1000
		forceSearch = True # used to avoid divergence
		gridSearchTime = 0.0
		# test
		self.check_period = 120
		c_target = 2
		expect_commit = [c_target for _ in expect_commit]

		lastSearch = check_cnt
		epochCnt = 0
		while True:
			time.sleep(1)
			if(not isfirst):
				check_cnt = check_cnt + 1
				# eta = 1.0 * pow(0.1, sum(step_for_each_worker) / 20000) / float(worker_num)
				if(forceSearch or (check_cnt - lastSearch >= epoch)):
					global start_time
					tmp_time = time.time()
					epochCnt += 1
					print("Epoch %d:" % (epochCnt))	
					self.allStop(ps_t)
					self.loadmodel(parameters)
					self.evaluation(10)
					last_loss = [self.global_loss]	
					last_loss_time = [time.time() - start_time]
					cur_reward, last_loss, last_loss_time = self.onlineSearch1Min(ps_t, c_target, last_loss, last_loss_time)
					next_reward, last_loss, last_loss_time = self.onlineSearch1Min(ps_t, c_target+1, last_loss, last_loss_time)

					# if next_reward c_target+1 is better, continue to search c_target + 2
					while(next_reward > cur_reward):
						cur_reward = next_reward
						c_target += 1 
						next_reward, last_loss, last_loss_time = self.onlineSearch1Min(ps_t, c_target+1, last_loss, last_loss_time)
					# now, current c_target is better
					expect_commit = [c_target for _ in expect_commit]

					self.allStart(ps_t)
					forceSearch = False
					lastSearch = check_cnt
					gridSearchTime += time.time() - tmp_time
					print("Time: %f\tAccu Search Time: %f\nOpt c_target: %f"%(time.time() - start_time, gridSearchTime, c_target))
					print ("USP:expect_commit %s\n" % (str(expect_commit)))
					self.record_info()
				# end if # epoch

				if(check_cnt % self.check_period == 0):
					self.commit_cnt = [self.commit_cnt[i] + delta_commit_cnt[i] for i in xrange(worker_num)]	
					## for record
					self.record_info()
					print("check_period: %d" % (check_cnt / self.check_period))

					delta_commit_cnt = [0 for _ in xrange(worker_num)]
					# evaluation
					global parameters
					last_global_loss = self.global_loss
					self.loadmodel(parameters)
					self.evaluation(10)
					if(last_global_loss != None and (self.global_loss - last_global_loss) / last_global_loss >= 0.3):
						forceSearch = True

					# send signal to child-thread, start to send expect commit #
					# print training_end_cnt
					if(self.global_loss < self.training_end):
						self.training_end_cnt += 1
					if(self.training_end_cnt >= 10):
						for i in xrange(worker_num):
							ps_t[i].isCheckOrStop = -1
						break			
				#end if
			#end if first
		# end while
		for i in xrange(worker_num):
			ps_t[i].join()
		self.f_speed.close()
		self.f_eval.close()
		self.f_hete.close()
		self.f_cmp_time.close()
		self.f_bandit_cnt.close()
		self.f_commit.close()

		self.f_global_loss.close()
		self.f_global_eval.close()
		print("USP_PS ends")




