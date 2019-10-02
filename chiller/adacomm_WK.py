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
from returnData import returnDataA

flags = tf.app.flags

flags.DEFINE_string('base_dir', '/home/net/hphu/chiller/', 'The path where log info will be stored')
flags.DEFINE_string('host', 'localhost', 'IP address of parameter server')
flags.DEFINE_integer('port_base', 14350, 'Start port for listening to workers')
flags.DEFINE_integer('class_num', 10, 'Training batch size ')
# wk
flags.DEFINE_integer('worker_index', 0, 'Index of worker')
flags.DEFINE_integer('max_steps', 1000000, 'Max steps each worker is allowed to run')
flags.DEFINE_float('sleep_time', 0, 'Specify the sleep_time')
flags.DEFINE_integer('batch_size', 64, 'mini-batch size')
flags.DEFINE_float('s', 10.0, 'Initial length of time between two commits for each worker')
FLAGS = flags.FLAGS


##############################################
#data type
TRTEMP = 0  #inlet return temperatrue, warmer one
TELE   = 1  #the chiller electricity
TFRATE = 2  #flow rate
TAGE   = 3  #the age of chiller
TCL    = 4  #the cooling load
TOT    = 5  #the outdoor temperature
TAELE  = 6  #the airside electricity                    
TPELE  = 7  #the pump electricity
TN     = 8  #total number of used features, i.e., feature No.0 - No.(TN-1) will be used
TCOP   = -1 #cop of individual chillers 
TPCOP  = -2 #cop of total chiller plant                                                 
TSTEMP = 100  #outlet supply temperature, cooler one, not supposed to use as feature

#Metrics
ER = 0
RMSE = 1
METRIC = RMSE
METRIC = ER
if METRIC == ER:
	print 'METRIC: ER'
elif METRIC == RMSE:
	print 'METRIC: RMSE'


USETOTALCOP = True
if USETOTALCOP:
	TCOP = TPCOP
	print 'MODE: COP_CHILLER_PLANT'
else:
	print 'MODE: COP_INDIVIDUAL_CHILLER'

TRAINK = 6.0 / 7
print 'TRAIN LENGTH (K):', TRAINK

#chiller number
#if changed, please modify the code in "typ >= TN" correspondingly
C1 = 0
C2 = 1
C3 = 2
C4 = 3
C5 = 4
CSN = 3 #the total number of chillers with the same type
CAN = 5 #the total number of chillers with all types
CN = 5 #the total number of chillers

print

def returnXy(mode=0, pX=None, index=None, K = TRAINK):
	temp = returnDataA(1, 0)[3]
	L = len(temp)
	start = 0
	ed = int(L * K)
	X = []
	y = []
	for ci in range(0, CAN, 1):
		for xi in range(0, ed-start, 1):
			X.append([])
		for fi in range(TN):
			arr = returnDataA(ci, fi)[3][start:ed]
			for xi in range(0, ed-start, 1):
				X[(ci)*(ed-start)+xi].append(float(arr[xi]))	
		cop = []
		cop = returnDataA(ci, TCOP)[3][start:ed]
		for yi in range(0, ed-start, 1):
			y.append(cop[yi])  
	return X, y

batch_size = FLAGS.batch_size
class_num = FLAGS.class_num
feature_size = TN

##############################################

class Model(object):
	def __init__(self):		
		self.class_num = FLAGS.class_num # 
		self.batch_size = FLAGS.batch_size # defined in cifar10.py
		self.No = FLAGS.worker_index


		self.base_dir = FLAGS.base_dir

		self.train_dir = self.base_dir + 'tmp/cifar10_train'
		self.max_steps = FLAGS.max_steps
		self.commit_cnt = 0 # record the total commit number 
		# # self.beta = [1 for _ in xrange(self.class_num)]
		self.loss_cnt = [0.0 for _ in xrange(self.class_num)]
		self.class_cnt = [0 for _ in xrange(self.class_num)]

		## for prediction
		self.predict_cnt = [0 for _ in xrange(self.class_num)]
		self.predict_rst = [0 for _ in xrange(self.class_num)]
		self.eval_rst = [0.0 for _ in xrange(self.class_num + 1)] # last elem is the overall accuracy
		
		## for ada
		self.local_c = 0
		self.min_c = 0
		self.s = FLAGS.s

		self.f_log = open(self.base_dir + 'wk_%d_ada.txt' % (self.No), 'w')
		self.f_pre = open(self.base_dir + 'wk_%d_ada_pred.txt' % (self.No), 'w')
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
		# declare the batch_size, placeholder, variable.
		self.x_data = tf.placeholder(shape=[None, feature_size], dtype=tf.float32)
		self.y_data = tf.placeholder(shape=[None], dtype=tf.int32)
		y_target = tf.transpose(tf.one_hot(self.y_data, class_num))
		# y_target = tf.placeholder(shape=[class_num, None], dtype=tf.float32)
		self.prediction_grid = tf.placeholder(shape=[None, feature_size], dtype=tf.float32)
		b = tf.Variable(tf.random_normal(shape=[class_num, batch_size]))

		# declare the Gaussian kernel.
		gamma = tf.constant(-10.0)
		dist = tf.reduce_sum(tf.square(self.x_data), 1)
		dist = tf.reshape(dist, [-1, 1])
		sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(self.x_data, tf.transpose(self.x_data)))), tf.transpose(dist))
		my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

		def reshape_matmul(mat):
			v1 = tf.expand_dims(mat, 1)
			v2 = tf.reshape(v1, [class_num, batch_size, 1])
			return(tf.matmul(v2, v1))

		# compute the dual loss function.
		model_output = tf.matmul(b, my_kernel)
		first_term = tf.reduce_sum(b)

		b_vec_cross = tf.matmul(tf.transpose(b), b)
		y_target_cross = reshape_matmul(y_target)
		second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)), [1, 2])
		self.loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))
		tf.summary.scalar('loss', self.loss)

		# create the prediction kernel
		rA = tf.reshape(tf.reduce_sum(tf.square(self.x_data), 1), [-1, 1])
		rB = tf.reshape(tf.reduce_sum(tf.square(self.prediction_grid), 1), [-1, 1])
		pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(self.x_data, tf.transpose(self.prediction_grid)))), tf.transpose(rB))
		pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

		# create the prediction
		prediction_output = tf.matmul(tf.multiply(y_target, b), pred_kernel)
		self.prediction = tf.argmax(prediction_output - tf.expand_dims(tf.reduce_mean(prediction_output, 1), 1), 0)
		accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(y_target, 0)), tf.float32))

		# declare the optimizer function
		my_opt = tf.train.GradientDescentOptimizer(0.001)
		self.train_op = my_opt.minimize(loss=self.loss, global_step=self.global_step)
		self.init = tf.global_variables_initializer()
		self.merged = tf.summary.merge_all() 	

		self.sess = tf.Session()
		self.sess.run(self.init)
		if tf.gfile.Exists(self.base_dir + 'board'):
			tf.gfile.DeleteRecursively(self.base_dir + 'board')
		tf.gfile.MakeDirs(self.base_dir + 'board')
		self.train_writer = tf.summary.FileWriter(self.base_dir + '/board/train', self.sess.graph)
		self.eval_writer = tf.summary.FileWriter(self.base_dir + '/board/eval', self.sess.graph)

	def prepare_data(self):
		# prepare train and test data
		# tX, ty = returnXy(0)
		# pX, py = returnXy(1)
		allx, _ally = returnXy(6)
		def classify(ylist):
			mu = np.mean(ylist)
			sigma = np.var(ylist)
			def normal(bins):
			    return (1.0 / np.sqrt(sigma * 2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma) )
			ref = normal(mu) / float(FLAGS.class_num / 2)
			return np.asarray([int(FLAGS.class_num - 1 - math.floor(normal(y) / ref)) if y > mu else int(math.floor(normal(y) / ref)) for y in ylist])
		ally = classify(_ally)

		train_indices = np.random.choice(len(allx), int(round(len(allx)*TRAINK)), replace=False)
		test_indices = np.array(list(set(range(len(allx))) - set(train_indices)))
		tx = np.array(allx)[train_indices]
		ty = np.array(ally)[train_indices]
		px = np.array(allx)[test_indices]
		py = np.array(ally)[test_indices]

		# class for managing data
		class dataset(object):
			"""docstring for dataset"""
			def __init__(self, x, y):
				self.x = x
				self.y = y
				self.index = 0
				self.len = len(x)
				self.iteration_cnt = 0
				print self.len
			def next_pair(self):
				bx = self.x[self.index]
				by = self.y[self.index]
				self.index += 1
				if(self.index >= self.len):
					self.index = 0
					self.iteration_cnt += 1
				print("\nClient %d" % self.iteration_cnt)
				return bx, by
		# convert raw data to class "dataset"
		def returnBatchData(x, y):
			# shuffle
			zipTXTy = zip(x, y)
			np.random.shuffle(zipTXTy)
			shuffle_tx, shuffle_ty = zip(*zipTXTy)
			# for reshape
			batch_cnt = len(x) / (batch_size * TN)
			input_len = batch_cnt * (batch_size * TN)
			_set_x = np.asarray(shuffle_tx[:input_len]).reshape(-1, batch_size, TN)
			_set_y = np.asarray(shuffle_ty[:input_len]).reshape(-1, batch_size)
			# return set
			return dataset(_set_x, _set_y)
		self.train_set = returnBatchData(tx, ty)
		self.test_set = returnBatchData(px, py)

	def fetch_data(self, Train = True, Eval = True, Test = True):
		train_X = train_Y = eval_X = eval_Y = test_X = test_Y = None

		if(Train):
			train_X, train_Y = self.train_set.next_pair()
		if(Eval):
			eval_X, eval_Y = self.test_set.next_pair()
		if(Test):
			test_X, test_Y = self.test_set.next_pair()
		return train_X, train_Y, eval_X, eval_Y, test_X, test_Y
	
	def register(self):
		self.X = self.x_data
		self.Y = self.y_data
		self.feed_prediction = self.prediction_grid # input for prediction(optional)
		self.top_k_op = self.prediction



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
		self.register()
		time_after_a_commit = time.time()
		# while not self.sess.should_stop():
		_s = 0
		while self.local_c < self.max_steps:
			time_step_start = time.time()
			self.local_c += 1
			_s += 1
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
			print
			print("%d:Time per batch: %f" % (_s, hete_time))
			print("time of one commit: %f" % (time_one_step))
			print("time from last commit: %f" % (time_span))
			print
			''' commit to the PS '''
			if(_s >= self.s):
				self.process_commit(cur_time, cur_loss, hete_time, time_one_step, cur_step, time_span)
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
				_s = 0						
			# end commit
		# end for
		print('ada_WK{} stops training - Step:{}\tTime:{}'.format(self.No, self.local_c, cur_time))
		self.evaluation(cur_time, 100)

	def evaluation(self, cur_time = None, eval_time = 10):
		for _ in xrange(eval_time):
			_, _, _, _, test_X, test_Y = self.fetch_data(Train = False, Eval = False, Test = True)
			predictions = self.sess.run(
				self.top_k_op, 
				feed_dict={
					self.X: test_X, 
					self.Y: test_Y,
					self.feed_prediction: test_X})
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

	def process_commit(self, cur_time, cur_loss, hete_time, time_one_step, cur_step, time_span):
		self.commit_cnt += 1
		self.skt.sendall('Commit')
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
		l = "%20.15f, %20.15f, %d, %d, %20.15f, %20.15f" % (hete_time, cur_loss, self.local_c, cur_step, time_one_step, time_span)
		self.skt.sendall(l)

		## Receive the overall parameters from the PS
		for i in xrange(len(self.parameter)):
			l = self.recvmsg()
			# communication_size += sys.getsizeof(l)
			self.parameter[i] = np.array(self.str2list(l)).reshape(self.para_shape[i])

		# receive 
		float(self.skt.recv(self.buffsize))
		# self.s = 
		
		# update the local mode
		# self.saver.save(self.sess, "./tmp/check.ckpt")
		index = 0
		for v in tf.trainable_variables():
			tt = time.time()
			v.load(self.parameter[index], self.sess)
			# self.sess.run(assign_op)
			print("		-- time: %f\tsize: %f" %(time.time() - tt, sys.getsizeof(self.parameter[index])))
			index += 1
		# print("ada_WK{}:Commit {} finished".format(self.No, self.commit_cnt))
		print("***************\nada_WK{}:Commit {} finished\n***********".format(self.No, self.commit_cnt))

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

		# change the role from sender to receiver
		self.skt.send("start receive msg")
		# receive target commit No 
		self.s = float(self.skt.recv(self.buffsize))
		print("current s: %f" % (self.s))


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


