# import tensorflow as st
import strain as st
import tensorflow as tf # if we use the latest version tensorflow, do not need to import tensorflow here
import os 
import math
import random
import numpy as np 

flags = tf.app.flags
# both worker and ps
flags.DEFINE_string('base_dir', './', 'The path where log info will be stored')
flags.DEFINE_string('host', 'localhost', 'IP address of parameter server')
flags.DEFINE_integer('port_base', 14400, 'Start port for listening to workers')
flags.DEFINE_string('job_name', '', 'Either ps or worker')
flags.DEFINE_integer('class_num', 10, 'Training batch size ')
flags.DEFINE_float('check_period', 60.0, 'Length of time between two checkpoints')
# wk
flags.DEFINE_integer('worker_index', 0, 'Index of worker')
flags.DEFINE_float('base_time_step', 20.0, 'Initial length of time between two commits for each worker')
flags.DEFINE_integer('max_steps', 1000000, 'Max steps each worker is allowed to run')
flags.DEFINE_float('sleep_time', 0, 'Specify the sleep_time')
flags.DEFINE_integer('batch_size', 8, 'mini-batch size')
# ps
flags.DEFINE_integer('worker_num', 1, 'Total number of workers')
flags.DEFINE_float('band_width_limit', None, 'Specify the constrait of bandwidth in the form of x M/s')
flags.DEFINE_float('training_end', 0, 'When loss is smaller than this, end training')

# flags.DEFINE_string('data_dir', '/tmp/mnist-data', 'Directory  for storing mnist data')

''' defined in file cifar.py
#
st.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
st.app.flags.DEFINE_string('data_dir', base_dir + 'tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
st.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
'''
# wk for rail
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_integer('filter_size', 1, 'filter_size')
flags.DEFINE_integer('feature_size', 10, 'feature_size')

FLAGS = flags.FLAGS

sess = None
fetch_data = None
merged = None
train_op = None
loss = None
global_step = None
X = None
Y = None
top_k_op = None

train_set = None
test_set = None

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



def build_model():
	global sess, fetch_data, merged, train_op, loss, global_step, X, Y, top_k_op
	## build the model
	# st.Graph().as_default()	
	global_step = tf.train.get_or_create_global_step()
	tf.summary.scalar('global_step', global_step)

	hidden = 128
	X = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.feature_size])
	Y = tf.placeholder(tf.int32, [FLAGS.batch_size])  # sparse_softmax... need labels of int32 or int32
	tf.summary.histogram('X_input', X)

	X_input = tf.reshape(X, [1, FLAGS.batch_size, FLAGS.feature_size])
	basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden, forget_bias=1.0)
	rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X_input, dtype = tf.float32)


	with tf.variable_scope('local3') as scope:
		# Move everything into depth so we can perform a single matrix multiply.
		reshape = tf.reshape(rnn_output, [-1, hidden])
		weights = _variable_with_weight_decay('weights', shape=[hidden, 384],
								stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
		local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

	dropout = tf.layers.dropout(inputs=local3, rate=0.4)
	stacked_outputs = tf.layers.dense(dropout, units = FLAGS.class_num)
	# outputs = tf.reshape(stacked_outputs, [-1, FLAGS.batch_size, class_num])
	outputs = tf.nn.softmax(stacked_outputs, name = "softmax_outputs")

	# Calculate predictions.
	top_k_op = tf.nn.in_top_k(outputs, Y, 1)

	cross_entropy = st.sparse_softmax_cross_entropy_with_logits(logits = outputs, labels = Y) # return, of the same shape as labels
	# st.summary.histogram('cross_entropy', cross_entropy)
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	tf.add_to_collection('losses', cross_entropy_mean)
	loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
	tf.summary.scalar('loss', loss)

	decay_steps = 1000
	LEARNING_RATE_DECAY_FACTOR = 0.9
	lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
	tf.summary.scalar('learning_rate', lr)

	optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
	train_op = optimizer.minimize(loss, global_step = global_step)
	init = tf.global_variables_initializer()

	# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
	merged = tf.summary.merge_all()	

	sess = tf.Session()
	sess.run(init)

	if tf.gfile.Exists(FLAGS.base_dir + 'board'):
		tf.gfile.DeleteRecursively(FLAGS.base_dir + 'board')
	tf.gfile.MakeDirs(FLAGS.base_dir + 'board')
	### ###################### emd model

	## below for input
	prepareData()

def prepareData(): 
	global train_set, test_set
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
		print filename
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

	train_set = dataset(input_data[train_indices], input_labels[train_indices])
	test_set = dataset(input_data[test_indices], input_labels[test_indices])

def fetch_data(Train = True, Eval = True, Test = True):
	global train_set, test_set
	train_X = train_Y = eval_X = eval_Y = test_X = test_Y = None

	if(Train):
	    train_X, train_Y = train_set.next_pair()
	if(Eval):
	    eval_X, eval_Y = test_set.next_pair()
	if(Test):
	    test_X, test_Y = test_set.next_pair()
	return train_X, train_Y, eval_X, eval_Y, test_X, test_Y

def process_worker():
	global sess, fetch_data, merged, train_op, loss, global_step, X, Y, top_k_op
	# train_dir = FLAGS.base_dir + 'tmp/cifar10_train'
	# preprocess
	# if tf.gfile.Exists(train_dir):
	# 	tf.gfile.DeleteRecursively(train_dir)
	# tf.gfile.MakeDirs(train_dir)

	strain_wk = st.STRAIN_WK(
			_worker_index = FLAGS.worker_index, 
			_check_period = FLAGS.check_period, 
			_init_base_time_step = FLAGS.base_time_step, 
			_max_steps = FLAGS.max_steps, 
			_batch_size = FLAGS.batch_size, 
			_class_num = FLAGS.class_num,
			_base_dir = FLAGS.base_dir,
			_host = FLAGS.host,
			_port_base = FLAGS.port_base)
	# cross_entropy is only related to loss_cnt, maybe it can be deleted: huhanpeng
	build_model()
	strain_wk.register(
		_session = sess,
		_func_fetch_data = fetch_data,
		_merged = merged,
		_train_op = train_op,
		_loss = loss,
		_global_step = global_step,
		_feed_X = X,
		_feed_Y = Y,
		_top_k_op = top_k_op)

	strain_wk.run(simulate = FLAGS.sleep_time)	
	sess.close()

def process_server():
	global sess, fetch_data, merged, train_op, loss, global_step, X, Y, top_k_op
	strain_ps = st.STRAIN_PS(
		_total_worker_num = FLAGS.worker_num,
		_check_period = FLAGS.check_period, 
		_class_num = FLAGS.class_num,
		_base_dir = FLAGS.base_dir,
		_host = FLAGS.host,
		_port_base = FLAGS.port_base,
		_band_width_limit = FLAGS.band_width_limit,
		_training_end = FLAGS.training_end
		)
	build_model()
	strain_ps.register(
		_session = sess,
		_func_fetch_data = fetch_data,
		_merged = merged,
		_train_op = train_op,
		_loss = loss,
		_global_step = global_step,
		_feed_X = X,
		_feed_Y = Y,
		_top_k_op = top_k_op)
	strain_ps.run()

def main(argv):
	if(FLAGS.job_name == 'ps'):
		process_server()
	elif(FLAGS.job_name == 'worker'):
		process_worker()
	else:
		print("ArgumentError:argument <job_name> must be ps or worker")
	


if __name__ == "__main__":
  	tf.app.run()


