import threading
from socket import *
import time 
import json  # for transport structured data
import tensorflow as tf

flags = tf.app.flags

# both worker and ps
flags.DEFINE_string('base_dir', '/home/net/hphu/chiller/', 'The path where log info will be stored')
flags.DEFINE_string('host', 'localhost', 'IP address of parameter server')
flags.DEFINE_integer('port_base', 14350, 'Start port for listening to workers')
flags.DEFINE_integer('class_num', 10, 'Training batch size ')
flags.DEFINE_float('check_period', 60.0, 'Length of time between two checkpoints')
flags.DEFINE_float('s', 40.0, 'Initial length of time between two commits for each worker')

# ps
flags.DEFINE_integer('worker_num', 3, 'Total number of workers')
flags.DEFINE_float('band_width_limit', None, 'Specify the constrait of bandwidth in the form of x M/s')
flags.DEFINE_float('training_end', 0, 'When loss is smaller than this, end training')

FLAGS = flags.FLAGS


# global variable
worker_num = FLAGS.worker_num
class_num = FLAGS.class_num

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

base_dir = FLAGS.base_dir
print base_dir
f_log = open(base_dir + 'ps_log_alter_ssp.txt', 'w')
f_hete = open(base_dir + 'ps_hete_alter_ssp.txt', 'w')
f_eval = open(base_dir + 'ps_eval_alter_ssp.txt', 'w')
f_cmp_time = open(base_dir + 'ps_cmp_time_alter_ssp.txt', 'w')
f_speed = open(base_dir + 'ps_speed_alter_ssp.txt', 'w')

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
		global loss_for_each_worker, hete_for_each_worker, step_for_each_worker, comp_time_for_each_worker, speed_for_each_worker
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
		comp_time_for_each_worker[self.index] += float(msg[4])
		speed_for_each_worker[self.index] = float(msg[4])

	def return_commit(self):
		# global parameters, class_cnt, gamma_cnt
		# self.skt_client.recv(self.buffsize)
		# calculate and send beta

		global clock_cnt
		# if fastest worker is faster that the slowest worker by more than s clock, hang up
		hung_cnt = 0
		while True:
			min_c = min(clock_cnt)
			if(clock_cnt[self.index] - min_c < self.s or self.isCheckOrStop == -1):
				break
			time.sleep(1)
			hung_cnt += 1
			# print("Worker %d has been blocked for %d seconds" % (self.index, hung_cnt))

		if(self.isCheckOrStop == -1):
			print("Worker %d check stop break" % (self.index))
		else:
			print("Worker %d has been blocked for %d seconds" % (self.index, hung_cnt))

		self.total_hung_cnt += hung_cnt

		# K = 1
		# reciprocal_cnt = [float(1) / float(x+1) for x in class_cnt] # x+1??: huhanpeng
		# max_re_cnt = max(reciprocal_cnt)
		# beta = [K * float(x) / float(max_re_cnt) for x in reciprocal_cnt]
	
		# send overall parameters
		for i in xrange(len(parameters)):
			self.sendmsg(str(parameters[i]))
		#send overall min class index
		self.skt_client.sendall(str(min_c))

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

def server():
	global worker_num, class_num, commit_cnt
	global f_log, f_eval, f_hete
	global loss_for_each_worker, hete_for_each_worker, step_for_each_worker, comp_time_for_each_worker, speed_for_each_worker
	global eval_rst	
	check_per_cnt = FLAGS.check_period
	check_cnt = 0
	training_end_cnt = 0
	# K2 = 2

	ps_t = [PSThread(i) for i in xrange(worker_num)]
	for i in xrange(worker_num):
		ps_t[i].start()
 
	while True:
		check_cnt = check_cnt + 1
		time.sleep(1)
		# print("sleep one second %d " % check_cnt)
		if(check_cnt >= check_per_cnt):						
			# send signal to child-thread, start to send expect commit #
			# for record
			cur_time = time.time() - start_time
			last_avg_loss = average(loss_for_each_worker)

			f_log.write('%020.10f, %020.10f, %030.20f, %020.10f\n' % 
					(cur_time, average(step_for_each_worker), last_avg_loss, average(commit_cnt)))
			f_hete.write('%s\n' % (str(hete_for_each_worker)))
			f_eval.write('%s\n' % str(eval_rst))
			f_cmp_time.write('%s\n' % str(comp_time_for_each_worker))
			f_speed.write('%s\n' % str(speed_for_each_worker))
			f_log.flush()
			f_hete.flush()
			f_eval.flush()
			f_cmp_time.flush()
			f_speed.flush()
			# assure expect_commit >= 1, at least once
			print ("ALTER_SSP:commit_cnt %s" % str(commit_cnt))	

			if(last_avg_loss < FLAGS.training_end):
				training_end_cnt += 1
			if(training_end_cnt == 10):
				for i in xrange(worker_num):
					ps_t[i].isCheckOrStop = -1
				break
			else:
				for i in xrange(worker_num):
					ps_t[i].isCheckOrStop = 1
				check_cnt = 0		
		# 	endif
	# end while
	for i in xrange(worker_num):
			ps_t[i].join()
	print("ALTER_PS ends")
	f_log.close()
	f_hete.close()
	f_eval.close()
	f_cmp_time.close()
	f_speed.close()




if __name__ == "__main__":
	server()

