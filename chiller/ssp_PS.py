import threading
from socket import *
import time 
import json  # for transport structured data

import os 
import math

import tensorflow as tf
import numpy as np
import sys as sys
from returnData import returnDataA

flags = tf.app.flags
flags.DEFINE_string('base_dir', '/home/net/hphu/chiller/', 'The path where log info will be stored')
flags.DEFINE_string('host', 'localhost', 'IP address of parameter server')
flags.DEFINE_integer('port_base', 14320, 'Start port for listening to workers')
flags.DEFINE_integer('class_num', 10, 'Training batch size ')
flags.DEFINE_float('check_period', 60.0, 'Length of time between two checkpoints')
flags.DEFINE_float('s', 40.0, 'Initial length of time between two commits for each worker')
flags.DEFINE_bool('Fixed', False, 'wether fix the commit rate')
# ps
flags.DEFINE_integer('worker_num', 3, 'Total number of workers')
flags.DEFINE_float('band_width_limit', None, 'Specify the constrait of bandwidth in the form of x M/s')
flags.DEFINE_float('training_end', 0, 'When loss is smaller than this, end training')
flags.DEFINE_integer('batch_size', 64, 'mini-batch size')

FLAGS = flags.FLAGS

### for model 
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
feature_size = TN
#######################

worker_num = FLAGS.worker_num
class_num = FLAGS.class_num

TMP_LIST_UP_BOUND = 3
LOSS_LIST_UP_BOUND = 10
STEP_SIZE_REDUCE_RATE = 0.9

clock_cnt = [0 for _ in xrange(worker_num)]
push_cnt = [0 for _ in xrange(worker_num)]
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
global_loss = 0.0


mutex = threading.Lock()

class PSThread(threading.Thread):	
	def __init__(self, worker_index):
		threading.Thread.__init__(self)
		self.host = FLAGS.host
	 	self.port_base = FLAGS.port_base
	 	self.buffsize = 1024
	 	self.index = worker_index
	 	self.s = FLAGS.s
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

		totalsize = 0.0
		for x in parameters:
			# print len(x)
			totalsize += sys.getsizeof(x)
		print("The size of model: %fMB" % (totalsize / (1024.0 * 1024.0)))

		# start to listen to the port
		global push_cnt
		while self.go_on:
			msg = self.skt_client.recv(self.buffsize)
			if('Push' in msg):
				push_cnt[self.index] += 1
				self.skt_client.sendall("start push")
				self.recv_push()
			elif('Pull' in msg):
				self.return_pull()
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

	def recv_push(self):
		global parameters, clock_cnt
		global loss_for_each_worker, hete_for_each_worker, step_for_each_worker, comp_time_for_each_worker
		# recv and update parameters
		for i in xrange(len(parameters)):
			new_l = self.str2list(self.recvmsg())
			parameters[i] = [parameters[i][x] + new_l[x] / float(worker_num) for x in xrange(len(new_l))]
			# parameters[i] = [parameters[i][x] + new_l[x] for x in xrange(len(new_l))]
		msg = self.recvmsg().split(',')
		# record the loss and clock 

		hete_for_each_worker[self.index] = float(msg[0])
		loss_for_each_worker[self.index] = float(msg[1])
		clock_cnt[self.index] = int(msg[2])
		step_for_each_worker[self.index] = float(msg[3])
		comp_time_for_each_worker[self.index] += float(msg[4])
		speed_for_each_worker[self.index] = float(msg[4])

	def return_pull(self):
		# global parameters, class_cnt, gamma_cnt
		# self.skt_client.recv(self.buffsize)
		# calculate and send beta
		global clock_cnt
		# if fastest worker is faster that the slowest worker by more than s clock, hang up
		hung_cnt = 0
		tmp_time = time.time()
		while True:
			min_c = min(clock_cnt)
			if(clock_cnt[self.index] - min_c < self.s or self.isCheckOrStop == -1):
				break
			# print("Worker %d has been blocked for %d seconds" % (self.index, hung_cnt))
		hung_cnt = time.time() - tmp_time
		if(self.isCheckOrStop == -1):
			print("Worker %d check stop break" % (self.index))
		else:
			print("Worker %d has been blocked for %d seconds" % (self.index, hung_cnt))
		
		self.total_hung_cnt += hung_cnt
		# send overall parameters
		for i in xrange(len(parameters)):
			self.sendmsg(str(parameters[i]))
		#send overall min class index
		self.skt_client.sendall(str(min_c))

		# self.sendmsg(str(beta))
		# # calculate and send gamma
		# # self.sendmsg(str(gamma_cnt[self.index] / sum(gamma_cnt)))

	def check(self):
		self.skt_client.recv(self.buffsize)
		# check --- send some message
		if self.isCheckOrStop == 1:
			self.process_check()
		elif(self.isCheckOrStop == -1):
			self.skt_client.sendall('Stop')
			self.go_on = False
		else:
			self.skt_client.sendall('No')

	def process_check(self):
		global class_cnt, eval_rst
		self.skt_client.sendall('Check')
		# recv and update class cnt
		new_l = self.str2list(self.recvmsg())
		class_cnt = [class_cnt[x] + new_l[x] for x in xrange(len(new_l))]
		# recv and update loss cnt
		new_l = self.str2list(self.recvmsg())
		for x in xrange(len(eval_rst[self.index])):
			eval_rst[self.index][x] = new_l[x]
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

		self.f_log = open(self.base_dir + 'ps_log_ssp.txt', 'w')
		self.f_hete = open(self.base_dir + 'ps_hete_ssp.txt', 'w')
		self.f_eval = open(self.base_dir + 'ps_eval_ssp.txt', 'w')
		self.f_cmp_time = open(self.base_dir + 'ps_cmp_time_ssp.txt', 'w')
		self.f_speed = open(self.base_dir + 'ps_speed_ssp.txt', 'w')

		self.f_global_loss = open(self.base_dir + 'ps_global_loss_ssp.txt', 'w')
		self.f_global_eval = open(self.base_dir + 'ps_global_eval_ssp.txt', 'w')
		
		self.band_width_limit = FLAGS.band_width_limit

		self.predict_cnt = [0 for _ in xrange(class_num)]
		self.predict_rst = [0 for _ in xrange(class_num)]
		self.global_eval_rst = [0.0 for _ in xrange(class_num + 1)]

		self.build_model()

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
	
	def loadmodel(self):
		index = 0
		global parameters
		for v in tf.trainable_variables():
			shape = v.get_shape()
			load_v = np.array(parameters[index]).reshape(shape)
			v.load(load_v, self.sess)
			index += 1

	def evaluation(self, eval_time = 10):
		global class_num, step_for_each_worker, global_loss, start_time
		global_loss = 0.0
		cur_time = time.time() - start_time
		for _ in xrange(eval_time):
			_, _, _, _, test_X, test_Y = self.fetch_data(Train = False, Eval = False, Test = True)
			predictions, _loss = self.sess.run(
				[self.top_k_op, self.loss],
				feed_dict={
					self.X: test_X, 
					self.Y: test_Y,
					self.feed_prediction: test_X})
			# print(predictions)
			global_loss += _loss
			for i in xrange(len(test_Y)):
				self.predict_cnt[test_Y[i]] += 1
				if(predictions[i]):
					self.predict_rst[test_Y[i]] += 1
		global_loss = global_loss / float(eval_time)
		print("Time: %20.10f \tLoss: %f\tTotal accuracy: %20.19f\n" %\
			(cur_time, global_loss, sum(self.predict_rst) / float(sum(self.predict_cnt))))
		
		## calculate accuracy for each class
		self.global_eval_rst[-1] = sum(self.predict_rst) / float(sum(self.predict_cnt))
		for i in xrange(class_num):
			if(self.predict_cnt[i] == 0):
				self.global_eval_rst[i] = -1.0
			else:
				self.global_eval_rst[i] = float(self.predict_rst[i]) / self.predict_cnt[i]
		self.f_global_eval.write('%s\n' % str(self.global_eval_rst))
		self.f_global_loss.write('%020.10f, %020.10f, %020.10f, %030.20f\n' % 
							(cur_time, average(step_for_each_worker), sum(step_for_each_worker), global_loss))
		self.f_global_eval.flush()
		self.f_global_loss.flush()

		self.predict_cnt = [0 for _ in xrange(class_num)]
		self.predict_rst = [0 for _ in xrange(class_num)]

	def run(self):
		global worker_num, class_num, push_cnt
		global loss_for_each_worker, hete_for_each_worker, step_for_each_worker, comp_time_for_each_worker, speed_for_each_worker
		global eval_rst, start_time

		check_cnt = 0
		training_end_cnt = 0

		ps_t = [PSThread(i) for i in xrange(worker_num)]
		for i in xrange(worker_num):
			ps_t[i].start()

		self.prepare_data()	
		self.register()

		global isfirst
		while True:
			time.sleep(1)
			if(not isfirst):
				check_cnt = check_cnt + 1
				# print("sleep one second %d " % check_cnt)
				if(check_cnt >= self.check_period):						
					# send signal to child-thread, start to send expect commit #
					# for record
					cur_time = time.time() - start_time
					last_avg_loss = average(loss_for_each_worker)

					self.f_log.write('%020.10f, %020.10f, %020.10f, %030.20f, %020.10f\n' % 
							(cur_time, average(step_for_each_worker), sum(step_for_each_worker), last_avg_loss, average(push_cnt)))
					self.f_hete.write('%020.10f: %s\n' % (cur_time, str(hete_for_each_worker)))
					self.f_eval.write('%020.10f: %s\n' % (cur_time, str(eval_rst)))
					self.f_cmp_time.write('%020.10f: %s\n' % (cur_time, str(comp_time_for_each_worker)))
					self.f_speed.write('%020.10f: %s\n' % (cur_time, str(speed_for_each_worker)))
					self.f_log.flush()
					self.f_hete.flush()
					self.f_eval.flush()
					self.f_cmp_time.flush()
					self.f_speed.flush()
					# assure expect_commit >= 1, at least once
					print ("SSP:push_cnt %s" % str(push_cnt))

					self.loadmodel()
					self.evaluation(10)	

					global global_loss
					if(global_loss < FLAGS.training_end):
						training_end_cnt += 1
					if(training_end_cnt == 10):
						for i in xrange(worker_num):
							ps_t[i].isCheckOrStop = -1
						break
					else:
						for i in xrange(worker_num):
							ps_t[i].isCheckOrStop = 1
						check_cnt = 0		
				# endif >= check_period
			#end if not first
		# end while
		for i in xrange(worker_num):
				ps_t[i].join()
		print("SSP_PS ends")
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

