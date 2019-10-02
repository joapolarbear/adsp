import threading
from socket import *
import time 
import json  # for transport structured data

import os 
import math

import tensorflow as tf
import numpy as np
import copy
import math

flags = tf.app.flags

flags.DEFINE_integer('class_num', 10, 'Training batch size ')
flags.DEFINE_float('check_period', 60.0, 'Length of time between two checkpoints')
flags.DEFINE_integer('worker_num', 1, 'Total number of workers')
flags.DEFINE_string('host', '202.45.128.146', 'IP address of parameter server')
flags.DEFINE_integer('port_base', 14470, 'Start port for listening to workers')
flags.DEFINE_string('base_dir', '/home/net/hphu/rail/', 'The path where log info will be stored')
flags.DEFINE_float('training_end', 0, 'When loss is smaller than this, end training')

flags.DEFINE_float('band_width_limit', None, 'Specify the constrait of bandwidth in the form of x M/s')

flags.DEFINE_integer('batch_size', 8, 'mini-batch size')

# model rail
flags.DEFINE_integer('filter_size', 1, 'filter_size')
flags.DEFINE_integer('feature_size', 10, 'feature_size')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_float('s', 40.0, 'Initial length of time between two commits for each worker')
flags.DEFINE_bool('Fixed', False, 'wether fix the commit rate')

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


worker_num = FLAGS.worker_num
class_num = FLAGS.class_num

TMP_LIST_UP_BOUND = 3
LOSS_LIST_UP_BOUND = 10
STEP_SIZE_REDUCE_RATE = 0.9

clock_cnt = [0 for _ in xrange(worker_num)]
commit_cnt = [0 for _ in xrange(worker_num)]
# gamma_cnt = [0 for _ in xrange(worker_num)]
loss_for_each_worker= [0.0 for _ in xrange(worker_num)]
hete_for_each_worker = [0.0 for _ in xrange(worker_num)]
step_for_each_worker = [0 for _ in xrange(worker_num)]
comp_time_for_each_worker = [0 for _ in xrange(worker_num)]
speed_for_each_worker = [0 for _ in xrange(worker_num)]

class_cnt = [0 for _ in xrange(class_num)]
eval_rst = [[0.0 for _ in xrange(class_num + 1)] for _ in xrange(worker_num)]

# expect_commit = [0 for _ in xrange(worker_num)]

parameters = []
isfirst = True
start_time = time.time()

checkInCnt = 0
tau = 40.0

mutex = threading.Lock()

class PSThread(threading.Thread):	
	def __init__(self, worker_index):
		threading.Thread.__init__(self)
		self.host = FLAGS.host
	 	self.port_base = FLAGS.port_base
	 	self.buffsize = 1024
	 	self.index = worker_index
	 	self.isCheckOrStop = 0	
	 	self.go_on = True
	 	self.total_hung_cnt = 0

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
				parameters.append(self.str2list(msg))	
			# end while
		else:
			self.skt_client.sendall('No')
			self.skt_client.recv(self.buffsize)
			for i in xrange(len(parameters)):
				self.sendmsg(str(parameters[i]))
			self.skt_client.sendall('$')
		mutex.release()

		for x in parameters:
			print len(x)

		# start to listen to the port
		global commit_cnt
		while self.go_on:
			msg = self.skt_client.recv(self.buffsize)
			if('Commit' in msg):
				commit_cnt[self.index] += 1
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
		print("Worker %d has been totally blocked for %d seconds" % (self.index, self.total_hung_cnt))
		time.sleep(1)	
		# end while

	def recv_commit(self):
		global parameters, clock_cnt
		global loss_for_each_worker, hete_for_each_worker, step_for_each_worker, comp_time_for_each_worker
		# recv and update parameters
		for i in xrange(len(parameters)):
			new_l = self.str2list(self.recvmsg())
			parameters[i] = [parameters[i][x] + new_l[x] / float(worker_num) for x in xrange(len(new_l))]
			# parameters[i] = [parameters[i][x] + new_l[x] for x in xrange(len(new_l))]
		msg = self.skt_client.recv(self.buffsize).split(',')
		# record the loss and clock 

		hete_for_each_worker[self.index] = float(msg[0])
		loss_for_each_worker[self.index] = float(msg[1])
		clock_cnt[self.index] = int(msg[2])
		step_for_each_worker[self.index] = float(msg[3])
		comp_time_for_each_worker[self.index] += float(msg[5])
		speed_for_each_worker[self.index] = float(msg[4])

	def return_commit(self):
		# global parameters, class_cnt, gamma_cnt
		# self.skt_client.recv(self.buffsize)
		# calculate and send beta
		global clock_cnt
		# if fastest worker is faster that the slowest worker by more than s clock, hang up
		tmp_time = time.time()
		while True:
			if(max(clock_cnt) == min(clock_cnt) or self.isCheckOrStop == -1):
				break
			# print("Worker %d has been blocked for %d seconds" % (self.index, hung_cnt))
		hung_cnt = time.time() - tmp_time
		# if(self.isCheckOrStop == -1):
		# 	print("Worker %d check stop break" % (self.index))
		# else:
		# 	print("Worker %d has been blocked for %d seconds" % (self.index, hung_cnt))
		
		self.total_hung_cnt += hung_cnt
		# send overall parameters
		for i in xrange(len(parameters)):
			self.sendmsg(str(parameters[i]))
		#send overall min class index
		global tau
		self.skt_client.sendall(str(tau))

		# self.sendmsg(str(beta))
		# # calculate and send gamma
		# # self.sendmsg(str(gamma_cnt[self.index] / sum(gamma_cnt)))

	def check(self):
		self.skt_client.recv(self.buffsize)
		# check --- send some message
		if self.isCheckOrStop == 1:
			self.ps_process_check()
		elif(self.isCheckOrStop == -1):
			self.skt_client.sendall('Stop')
			self.go_on = False
		else:
			self.skt_client.sendall('No')

	def ps_process_check(self):
		# sync !!!
		global worker_num, checkInCnt
		mutex.acquire()
		checkInCnt += 1
		print("Worker %d arrives, checkInCnt: %d" % (self.index, checkInCnt))
		mutex.release()
		# after reveiving start signal, jump out following loop
		while(self.isCheckOrStop != 2):
			pass

		global class_cnt, eval_rst
		self.skt_client.sendall('Check')
		# recv and update class cnt
		new_l = self.str2list(self.recvmsg())
		class_cnt = [class_cnt[x] + new_l[x] for x in xrange(len(new_l))]
		# recv and update loss cnt
		new_l = self.str2list(self.recvmsg())
		for x in xrange(len(eval_rst[self.index])):
			eval_rst[self.index][x] = new_l[x]

		# change the role from receiver to sender
		self.skt_client.recv(self.buffsize)
		global tau
		self.skt_client.sendall(str(tau))

		self.isCheckOrStop = 0

def average(seq):
	return float(sum(seq)) / len(seq)

class PARA_SERVER(object):
	def __init__(self):
		global worker_num, class_num
		self.batch_size = FLAGS.batch_size # defined in cifar10.py
		self.class_num = class_num

		self.check_period = FLAGS.check_period
		self.max_frequency = 10	#per check_period
		self.base_dir = FLAGS.base_dir

		self.f_log = open(self.base_dir + 'ps_log_ada.txt', 'w')
		self.f_hete = open(self.base_dir + 'ps_hete_ada.txt', 'w')
		self.f_eval = open(self.base_dir + 'ps_eval_ada.txt', 'w')
		self.f_cmp_time = open(self.base_dir + 'ps_cmp_time_ada.txt', 'w')
		self.f_speed = open(self.base_dir + 'ps_speed_ada.txt', 'w')

		self.f_global_loss = open(self.base_dir + 'ps_global_loss_ada.txt', 'w')
		self.f_global_eval = open(self.base_dir + 'ps_global_eval_ada.txt', 'w')
		
		self.band_width_limit = FLAGS.band_width_limit

		self.predict_cnt = [0 for _ in xrange(class_num)]
		self.predict_rst = [0 for _ in xrange(class_num)]
		self.global_eval_rst = [0.0 for _ in xrange(class_num + 1)]

		self.global_loss = None

		self.build_model()

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
	
	def loadmodel(self, _parameters):
		index = 0
		for v in tf.trainable_variables():
			shape = v.get_shape()
			load_v = np.array(_parameters[index]).reshape(shape)
			v.load(load_v, self.sess)
			index += 1

	def evaluation(self, eval_time = 10):
		global class_num, step_for_each_worker, start_time
		self.global_loss = 0.0
		cur_time = time.time() - start_time
		for _ in xrange(eval_time):
			train_X, train_Y, eval_X, eval_Y, test_X, test_Y = self.fetch_data(Train = True, Eval = True, Test = True)
			predictions, _loss = self.sess.run(
				[self.top_k_op, self.loss],
				feed_dict={
					self.X: test_X, 
					self.Y: test_Y})
			# print(predictions)
			self.global_loss += _loss
			for i in xrange(len(test_Y)):
				self.predict_cnt[test_Y[i]] += 1
				if(predictions[i]):
					self.predict_rst[test_Y[i]] += 1
		self.global_loss = self.global_loss / float(eval_time)
		print("Time: %20.10f \tLoss: %f\tTotal accuracy: %20.19f\n" %\
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

	# stop all threads
	def allStop(self, ps_t):
		global worker_num, checkInCnt
		print("Start waiting, checkInCnt: %d" % checkInCnt)
		for i in xrange(worker_num):
			ps_t[i].isCheckOrStop = 1
		print("Waiting all stop")
		while(checkInCnt < worker_num):
			pass
		checkInCnt = 0

	# start all threads
	def allStart(self, ps_t):
		global worker_num
		for i in xrange(worker_num):
			ps_t[i].isCheckOrStop = 2

	def offlineSearchOneS(self, opt_tau, max_dec, tmp_tau, init_loss, backup_para, ps_t):
		global parameters
		global tau
		tau = tmp_tau
		cnt = 0
		self.allStart(ps_t)
		while(cnt <= self.check_period):
			cnt += 1
			time.sleep(1)
		self.allStop(ps_t)

		print("tau: %f" %(tau))
		self.loadmodel(parameters)
		self.evaluation(10)
		dec = init_loss - self.global_loss
		if(dec > max_dec):
			opt_tau = tmp_tau
			max_dec = dec
		print("current opt tau: %f" %(opt_tau))
		parameters = copy.deepcopy(backup_para)
		return opt_tau, max_dec

	def offlineSearchS(self, ps_t):
		global parameters
		backup_para = copy.deepcopy(parameters)
		print("Start Search:")
		# until all workers have reached, go on
		self.loadmodel(parameters)
		self.evaluation(10)
		init_loss = self.global_loss

		max_dec = - 100000.0
		opt_tau = None

		for tmp_tau in [1, 5, 10, 20, 40, 60, 80, 100]:
			opt_tau, max_dec = self.offlineSearchOneS(
				opt_tau, max_dec,
				tmp_tau,
				init_loss, backup_para,
				ps_t)

		return opt_tau, init_loss

	def record_info(self, last_avg_loss):
		global start_time, commit_cnt
		cur_time = time.time() - start_time
		global hete_for_each_worker, step_for_each_worker, eval_rst, comp_time_for_each_worker, speed_for_each_worker

		self.f_log.write('%020.10f, %020.10f, %020.10f, %030.20f, %020.10f\n' % 
							(cur_time, average(step_for_each_worker), sum(step_for_each_worker), last_avg_loss, average(commit_cnt)))
		self.f_hete.write('%020.10f: %s\n' % (cur_time, str(hete_for_each_worker)))
		self.f_eval.write('%020.10f: %s\n' % (cur_time, str(eval_rst)))
		self.f_cmp_time.write('%020.10f: %s\n' % (cur_time, str(comp_time_for_each_worker)))
		self.f_speed.write('%020.10f: %s\n' % (cur_time, str(speed_for_each_worker)))
		self.f_log.flush()
		self.f_hete.flush()
		self.f_eval.flush()
		self.f_cmp_time.flush()
		self.f_speed.flush()

	def run(self):
		global worker_num, class_num, commit_cnt
		global loss_for_each_worker, hete_for_each_worker, step_for_each_worker, comp_time_for_each_worker, speed_for_each_worker
		global eval_rst, start_time

		check_cnt = 0
		training_end_cnt = 0

		ps_t = [PSThread(i) for i in xrange(worker_num)]
		for i in xrange(worker_num):
			ps_t[i].start()

		forceSearch = True
		gridSearchTime = 0.0
		tau_0 = None
		loss_0 = None
		global isfirst, tau
		while True:
			time.sleep(1)
			if(not isfirst):
				check_cnt = check_cnt + 1
				# print("sleep one second %d " % check_cnt)
				if(forceSearch):				
					tmp_time = time.time()
					self.allStop(ps_t)
					self.prepare_data()
					if(not FLAGS.Fixed):
						print("Initial: ")
						tau_0, loss_0 = self.offlineSearchS(ps_t)
						tau = tau_0
					else:
						tau = FLAGS.s
					self.allStart(ps_t)
					forceSearch = False
					gridSearchTime += time.time() - tmp_time
					cur_time = time.time() - start_time
					print("Time: %f\tAccu Search Time: %f\nOpt s: %f"%(cur_time, gridSearchTime, tau))
					print('Time spent on computing during Search time\n%020.10f: %s\n\n' % (cur_time, str(comp_time_for_each_worker)))		
				
				if(check_cnt >= self.check_period):	

					# self.allStop(ps_t)					
					# send signal to child-thread, start to send expect commit #
					# for record
					cur_time = time.time() - start_time
					last_avg_loss = average(loss_for_each_worker)

					self.record_info(last_avg_loss)
					print("check_period: %d\tcommit_cnt: %s\tlast s:%f" % (check_cnt / self.check_period, str(commit_cnt), tau))

					global parameters
					self.loadmodel(parameters)
					self.evaluation(10)	
					if(self.global_loss < FLAGS.training_end):
						training_end_cnt += 1
					if(training_end_cnt >= 10):
						for i in xrange(worker_num):
							ps_t[i].isCheckOrStop = -1
						break

					if(not FLAGS.Fixed):
						# adacomm update rule
						tmp_tau = math.sqrt(self.global_loss / loss_0) * tau_0
						rule = None
						# if(True):
						if(tmp_tau < tau):
							tau = math.ceil(tmp_tau)
							rule = 1
						else:
							tau = math.ceil(tau * 0.5)
							rule = 2
						print("Updated s: %f\tUpdate rule: %d" % (tau, rule))
				# endif >= check_period
			#end if not first
		# end while
		for i in xrange(worker_num):
				ps_t[i].join()
		print("ada_PS ends")
		self.f_log.close()
		self.f_hete.close()
		self.f_eval.close()
		self.f_cmp_time.close()
		self.f_speed.close()

		self.f_global_loss.close()
		self.f_global_eval.close()



if __name__ == "__main__":
	server = PARA_SERVER()
	server.run()

