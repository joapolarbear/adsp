'''
 version 4:
'''
import tensorflow as tf 
import numpy as np 
import os 
import math
import random
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

from socket import *
import json  # for transport structured data

from datetime import datetime
import time
import sys

flags = tf.app.flags

flags.DEFINE_integer('worker_index', 0, 'Index of worker')
flags.DEFINE_float('s', 40.0, 'Initial length of time between two commits for each worker')
flags.DEFINE_float('sleep_time', 1.0, 'Specify the sleep_time')
flags.DEFINE_string('host', '202.45.128.146', 'IP address of parameter server')
flags.DEFINE_integer('port_base', 14270, 'Start port for listening to workers')

flags.DEFINE_string('base_dir', '/home/net/hphu/rail/', 'The path where log info will be stored')
flags.DEFINE_integer('max_steps', 1000000, 'Max steps each worker is allowed to run')
flags.DEFINE_integer('class_num', 10, 'Training batch size ')

flags.DEFINE_integer('batch_size', 8, 'mini-batch size')
# wk rail
flags.DEFINE_integer('filter_size', 1, 'filter_size')
flags.DEFINE_integer('feature_size', 10, 'feature_size')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')

FLAGS = flags.FLAGS
def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """ 
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

class Model(object):
	def __init__(self):		
		self.class_num = FLAGS.class_num # 
		self.batch_size = FLAGS.batch_size # defined in cifar10.py
		self.No = FLAGS.worker_index
		self.base_dir = FLAGS.base_dir

		self.train_dir = self.base_dir + 'tmp/cifar10_train'
		self.max_steps = FLAGS.max_steps
		self.pull_cnt = 0 # record the total commit number 
		# # self.beta = [1 for _ in xrange(self.class_num)]
		self.loss_cnt = [0.0 for _ in xrange(self.class_num)]
		self.class_cnt = [0 for _ in xrange(self.class_num)]

		## for prediction
		self.predict_cnt = [0 for _ in xrange(self.class_num)]
		self.predict_rst = [0 for _ in xrange(self.class_num)]
		self.eval_rst = [0.0 for _ in xrange(self.class_num + 1)] # last elem is the overall accuracy
		
		## for ssp
		self.local_c = 0
		self.min_c = 0
		self.s = FLAGS.s

		self.f_log = open(self.base_dir + 'wk_%d_ssp.txt' % (self.No), 'w')
		self.f_pre = open(self.base_dir + 'wk_%d_ssp_pred.txt' % (self.No), 'w')
		# self.mse = 10
		# self.cnt = 0
		# self.ep_period = 100

		# build the model
		self.build_model()	
		self.parameter = []  # a list of parameters, parameters are np.array
		self.para_shape = []

		# connect to the PS
		# host = "202.45.128.146"
		# port_base = 14300
		host = FLAGS.host
		port_base = FLAGS.port_base
		addr = (host, port_base + self.No)
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
			self.skt.sendall('Load parameters')
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


	def build_model(self):
		self.global_step = tf.train.get_or_create_global_step()
		tf.summary.scalar('global_step', self.global_step)

		hidden = 128
		self.X = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.feature_size])
		self.Y = tf.placeholder(tf.int32, [FLAGS.batch_size])  # sparse_softmax... need labels of int32 or int32
		tf.summary.histogram('X_input', self.X)

		X_input = tf.reshape(self.X, [1, FLAGS.batch_size, FLAGS.feature_size])
		basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden, forget_bias=1.0)
		rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X_input, dtype = tf.float32)

		with tf.variable_scope('local3') as scope:
			# Move everything into depth so we can perform a single matrix multiply.
			reshape = tf.reshape(rnn_output, [-1, hidden])
			weights = _variable_with_weight_decay('weights', shape=[hidden, 384],
										stddev=0.04, wd=0.004)
			biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
			local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

		dropout = tf.layers.dropout(inputs = local3, rate = 0.4)
		stacked_outputs = tf.layers.dense(dropout, units = FLAGS.class_num)
		# outputs = tf.reshape(stacked_outputs, [-1, FLAGS.batch_size, class_num])
		outputs = tf.nn.softmax(stacked_outputs, name = "softmax_outputs")

		# Calculate predictions.
		self.top_k_op = tf.nn.in_top_k(outputs, self.Y, 1)

		self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = outputs, labels = self.Y) # return, of the same shape as labels
		# tf.summary.histogram('cross_entropy', cross_entropy) 
		cross_entropy_mean = tf.reduce_mean(self.cross_entropy)
		tf.add_to_collection('losses', cross_entropy_mean)
		self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		tf.summary.scalar('loss', self.loss)

		decay_steps = 1000
		LEARNING_RATE_DECAY_FACTOR = 0.9
		lr = tf.train.exponential_decay(FLAGS.learning_rate,
	                                  self.global_step,
	                                  decay_steps,
	                                  LEARNING_RATE_DECAY_FACTOR,
	                                  staircase=True)
		tf.summary.scalar('learning_rate', lr)

		optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
		self.train_op = optimizer.minimize(self.loss, global_step = self.global_step)
		self.init = tf.global_variables_initializer()

		# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
		self.merged = tf.summary.merge_all()	

		self.sess = tf.Session()
		self.sess.run(self.init)
		if tf.gfile.Exists(self.base_dir + 'board'):
			tf.gfile.DeleteRecursively(self.base_dir + 'board')
		tf.gfile.MakeDirs(self.base_dir + 'board')
		self.train_writer = tf.summary.FileWriter(self.base_dir + '/board/train', self.sess.graph)
		self.eval_writer = tf.summary.FileWriter(self.base_dir + '/board/eval', self.sess.graph)

	def prepare_data(self):
		def all_files_in(targetDir):
			_files = []
			cur_list = os.listdir(targetDir)
			for i in range(len(cur_list)):
				path = os.path.join(targetDir, cur_list[i])
				if os.path.isdir(path):
					_files.extend(list_all_files(path))
				if os.path.isfile(path):
					_files.append(path)
			return _files

		targetDir = os.path.join(FLAGS.base_dir, 'data/data/')
		allFiles = all_files_in(targetDir)
		data = []
		for filename in allFiles:
			# print filename
			for line in open(filename, 'r'):
				data.extend(line.strip().split('\n|\t'))
			# print(data[:endrow])
		data = list(map(float, data))


		input_len = int(len(data)/FLAGS.filter_size) - FLAGS.feature_size
		batch_cnt = input_len / (FLAGS.batch_size)
		input_len = batch_cnt * (FLAGS.batch_size)
		x = [[data[(i + j) * FLAGS.filter_size] for j in xrange(FLAGS.feature_size)] for i in xrange(input_len)] # shape: input_len * feature_size
		y = [data[(i + FLAGS.feature_size) * FLAGS.filter_size] for i in xrange(input_len)] # shape: input_len * 1
		# shuffle
		zipXY = zip(x, y)
		np.random.shuffle(zipXY)
		shuffle_x, shuffle_y = zip(*zipXY)
		# ######################
		# grade y to several class
		# change to 10 classes
		# def classify(ylist):
		# 	max_y = np.max(shuffle_y)
		# 	min_y = np.min(shuffle_y)
		# 	_a = float(math.e ** FLAGS.class_num - 1) / float(max_y + 1 - min_y)
		# 	_b = 1 - _a * min_y
		# 	return np.asarray([int(math.floor(math.log((_a * y + _b)))) for y in ylist])
		def classify(ylist):
			mu = np.mean(ylist)
			sigma = np.var(ylist)
			def normal(bins):
				return (1.0 / np.sqrt(sigma * 2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma) )
			ref = normal(mu) / float(FLAGS.class_num / 2)
			return np.asarray([int(FLAGS.class_num - 1 - math.floor(normal(y) / ref)) if y > mu else int(math.floor(normal(y) / ref)) for y in ylist])

		def f_one_hot(index, num):
			assert index < num
			rnt = np.asarray([0 for _ in xrange(num)])
			rnt[index] = 1
			return rnt
		# yy = classify(y)
		# y_ = np.asarray([f_one_hot(i, class_num) for i in yy])
		y_ = classify(shuffle_y)

		# class for managing data
		class dataset(object):
			"""docstring for dataset"""
			def __init__(self, x, y):
				self.x = x
				self.y = y
				self.index = 0
				self.len = len(x)
				self.iteration_cnt = 0

			def next_pair(self):
				bx = self.x[self.index]
				by = self.y[self.index]
				self.index += 1
				if(self.index >= self.len):
					self.index = 0
					self.iteration_cnt += 1
					print("\nClient %d" % self.iteration_cnt)
				return bx, by

		input_data = np.asarray(shuffle_x).reshape(-1, FLAGS.batch_size, FLAGS.feature_size)
		input_labels = y_.reshape(-1, FLAGS.batch_size)
		TRAINK = 6.0/7
		train_indices = np.random.choice(batch_cnt, int(round(batch_cnt * TRAINK)), replace=False)
		test_indices = np.array(list(set(range(batch_cnt)) - set(train_indices)))

		self.train_set = dataset(input_data[train_indices], input_labels[train_indices])
		self.test_set = dataset(input_data[test_indices], input_labels[test_indices])

	def fetch_data(self, Train = True, Eval = True, Test = True):
		train_X = train_Y = eval_X = eval_Y = test_X = test_Y = None

		if(Train):
			train_X, train_Y = self.train_set.next_pair()
		if(Eval):
			eval_X, eval_Y = self.test_set.next_pair()
		if(Test):
			test_X, test_Y = self.test_set.next_pair()
		return train_X, train_Y, eval_X, eval_Y, test_X, test_Y


	''' label = -1. clear class_cnt
		or update corresponding class_cnt
	'''
	def update_class(self, _labels = None, _cross_entropy = None):
		if(_labels == None):
			self.class_cnt = [0 for _ in xrange(self.class_num)]
		else:
			for i in xrange(len(_labels)):
				self.class_cnt[_labels[i]] += 1
		# end if

	def str2list(self, s):
		return [float(x) for x in s.split('[')[1].split(']')[0].split(',')]


	def run(self):	
		self.prepare_data()
		time_after_a_commit = time.time()
		# while not self.sess.should_stop():
		while self.local_c < self.max_steps:
			time_step_start = time.time()
			self.local_c += 1
			time.sleep(FLAGS.sleep_time)

			train_X, train_Y, eval_X, eval_Y, test_X, test_Y = self.fetch_data(Train = True, Eval = True, Test = True)
			## train one mini batch	
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
			
			# self.update_class(train_Y)	# update class_cnt
			cur_time = time.time() - self.start_time
			self.train_writer.add_summary(train_summary, int(cur_time))	
			self.f_log.write('%020.10f, %-10d, %030.20f\n' % (cur_time, cur_step, cur_loss))
			self.f_log.flush()
			
			# evaluation
			if(self.local_c % 10 == 0):
				eval_summary, _, eval_loss= \
					self.sess.run([
						self.merged, 
						self.train_op, 
						self.loss], 
					feed_dict = {
						self.X: eval_X, 
						self.Y: eval_Y})
				cur_time = time.time() - self.start_time
				self.eval_writer.add_summary(eval_summary, int(cur_time))	
				print('SSP_WK{} - Step:{}\tTime:{}\n\tTrain loss: {}\tEval loss: {}'
					.format(
						self.No, 
						self.local_c, 
						cur_time,
						cur_loss, 
						eval_loss))

			# record the a time
			time_one_step = time.time() - time_step_start 
			time_span = time.time() - time_after_a_commit
	

			self.push(cur_loss, hete_time, time_one_step, cur_step)
			print("Time to train one batch: %f" % (hete_time))
			print("Time of one step (include sleep time): %f" % (time_one_step))
			print("time from last pull: %f" % (time_span))
			print("Time of one step (include push): %f" % (time.time() - time_step_start))
			print
			''' commit to the PS '''
			if(self.local_c - self.min_c >= self.s):
				self.pull()
				# check
				self.skt.sendall('Check?')
				msg = self.skt.recv(self.buffsize)
				print msg
				if('Check' in msg):
					self.process_check(cur_time, cur_step)
				elif('Stop' in msg):
					break
				# recalculate time_after_a_commit after one commit
				time_after_a_commit = time.time()							
			# end commit
		# end for
		print('SSP_WK{} stops training - Step:{}\tTime:{}'.format(self.No, self.local_c, cur_time))
		self.evaluation(cur_time, 100)

	def evaluation(self, cur_time = None, eval_time = 10):
		for _ in xrange(eval_time):
			_, _, _, _, test_X, test_Y = self.fetch_data(Train = False, Eval = False, Test = True)
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

	def push(self, cur_loss, hete_time, time_one_step, cur_step):
		self.skt.sendall('Push')
		self.skt.recv(self.buffsize)
		# Sum all local updates
		index = 0
		for v in tf.trainable_variables():
			cur_parameter = v.eval(session = self.sess)
			# Send the result multiplied with gamma to the PS		
			l = str((((cur_parameter - self.parameter[index]).flatten())).tolist())
			self.sendmsg(l)
			# self.skt.sendall(str((((cur_parameter - self.parameter[index]).flatten())).tolist()))
			self.parameter[index] = cur_parameter[:]
			index += 1
		# end for

		## send local clock to the PS
		# change the role from sender to receiver
		l = "%20.15f, %20.15f, %d, %d, %20.15f" % (hete_time, cur_loss, self.local_c, cur_step, time_one_step)
		self.sendmsg(l)

	def pull(self):
		self.skt.sendall('Pull')
		self.pull_cnt += 1

		## Receive the overall parameters from the PS
		for i in xrange(len(self.parameter)):
			l = self.recvmsg()
			# communication_size += sys.getsizeof(l)
			self.parameter[i] = np.array(self.str2list(l)).reshape(self.para_shape[i])

		# receive 
		self.min_c = int(self.skt.recv(self.buffsize))

		# update the local mode
		# self.saver.save(self.sess, "./tmp/check.ckpt")
		index = 0
		for v in tf.trainable_variables():
			tt = time.time()
			v.load(self.parameter[index], self.sess)
			# self.sess.run(assign_op)
			print("		-- time: %f\tsize: %f" %(time.time() - tt, sys.getsizeof(self.parameter[index])))
			index += 1
		# print("SSP_WK{}:Commit {} finished".format(self.No, self.pull_cnt))
		print("***************\nSSP_WK{}: Pull {} finished\n***********".format(self.No, self.pull_cnt))


	def process_check(self, cur_time, cur_step):
		## Evaluation
		self.evaluation(cur_time)

		# Send the class counters and average loss cnt to the PS
		self.sendmsg(str(self.class_cnt))
		print("class_cnt %d: %s" % (self.No, self.class_cnt))
		# ave_loss = [self.loss_cnt[i] / self.class_cnt[i] for i in xrange(self.class_num)]
		self.sendmsg(str(self.eval_rst))
		# whether to clear these counter ??? : huhanpeng
		self.update_class()


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

	# end sess
# end class

def main(argv):
	model = Model()
	model.run()
	
	

if __name__ == "__main__":
  	tf.app.run()


