import tensorflow as tf
import strain as st
import cifar10_forImageNet as cifar10
import numpy as np
import vgg16
import os
# import vgg as vgg2
# slim = tf.contrib.slim

flags = tf.app.flags

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

# both worker and ps
flags.DEFINE_string('base_dir', './', 'The path where log info will be stored')
flags.DEFINE_string('host', 'localhost', 'IP address of parameter server')
flags.DEFINE_integer('port_base', 14200, 'Start port for listening to workers')
flags.DEFINE_string('job_name', '', 'Either ps or worker')
flags.DEFINE_integer('class_num', 10, 'Training batch size ')
flags.DEFINE_float('check_period', 60.0, 'Length of time between two checkpoints')
flags.DEFINE_integer('s', 40, 'threshold')
# wk
flags.DEFINE_integer('worker_index', 0, 'Index of worker')
flags.DEFINE_float('base_time_step', 40.0, 'Initial length of time between two commits for each worker')
flags.DEFINE_integer('max_steps', 1000000, 'Max steps each worker is allowed to run')
flags.DEFINE_float('sleep_time', 0, 'Specify the sleep_time')
# ps
flags.DEFINE_integer('worker_num', 1, 'Total number of workers')
flags.DEFINE_float('band_width_limit', None, 'Specify the constrait of bandwidth in the form of x M/s')
flags.DEFINE_float('training_end', 0, 'When loss is smaller than this, end training')
flags.DEFINE_float('epsilon', 0.3, 'Epsilon')


flags.DEFINE_float(
    'num_epochs_per_decay', 2.5,
    'Number of epochs after which learning rate decays. Note: this flag counts '
    'epochs per clone but aggregates per sync replicas. So 1.0 means that '
    'each clone will go over full epoch individually, but replicas will go '
    'once across all replicas.')
flags.DEFINE_float(
    'learning_rate_decay_factor', 0.1, 'Learning rate decay factor.')
flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

FLAGS = flags.FLAGS

sess = None
fetch_data = None
merged = None
train_op = None
loss = None
global_step = None
images = None
labels = None
top_k_op = None

train_images = None
train_labels = None
eval_images = None
eval_labels = None
test_images = None
test_labels = None

vgg = None
logits = None

train_writer=None


batch_size = 5
height, width = 224, 224
num_classes = 1000


def build_model():
	global sess, fetch_data, merged, train_op, loss, global_step, images, labels, top_k_op, train_images, train_labels, eval_images, eval_labels, test_images, test_labels
	global vgg, logits
	# build the model
	"""Train CIFAR-10 for a number of steps."""
	# tf.Graph().as_default()	
	global_step = tf.train.get_or_create_global_step()
	tf.summary.scalar('global_step', global_step)

	# '''
	# Get images and labels for CIFAR-10.
	# Force input pipeline to CPU:0 to avoid operations sometimes ending up on
	# GPU and resulting in a slow down.
	with tf.device('/cpu:0'):
		train_images, train_labels = cifar10.distorted_inputs()

	## Evaluation
	with tf.device('/cpu:0'):
		eval_images, eval_labels = cifar10.inputs(eval_data = True)

	## Test
	with tf.device('/cpu:0'):
		test_images, test_labels = cifar10.inputs(eval_data = False)
	# '''

	images = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 3])
	labels = tf.placeholder(dtype=tf.int32, shape=[None])

	# Build a Graph that computes the logits predictions from the
	# inference model.
	vgg = vgg16.vgg16(images)
	logits = vgg.probs

	# with slim.arg_scope(vgg2.vgg_arg_scope()):
	# 	logits, end_points = vgg2.vgg_19(images)
	# logits = cifar10.inference(images)

	# Calculate predictions.
	top_k_op = tf.nn.in_top_k(logits, labels, 1)

	'''# Calculate loss. huhanpeng: don't consider regularization
	loss, cross_entropy = cifar10.loss(vgg.probs, labels)
	train_op = cifar10.train(loss, global_step)
	'''
	# learning_rate = 0.01
	# decay_steps = 100000
	# learning_rate = tf.train.exponential_decay(0.01,
	# 					global_step,
	# 					decay_steps,
	# 					FLAGS.learning_rate_decay_factor,
	# 					staircase=True,
	# 					name='exponential_decay_learning_rate')
	# optimizer = tf.train.MomentumOptimizer(
	# 	learning_rate,
	# 	momentum=FLAGS.momentum,
	# 	name='Momentum')

	loss = cifar10.loss(logits, labels)
	train_op = cifar10.train(loss, global_step)
	config = None

	# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=vgg.probs))
	# train_op = optimizer.minimize(loss)
	# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	# cpu_num = int(os.environ.get('CPU_NUM', 1))
	# config = tf.ConfigProto(device_count={"CPU": cpu_num},
	# 	inter_op_parallelism_threads = cpu_num,
	# 	intra_op_parallelism_threads = cpu_num,
	# 	log_device_placement=True)
	# config.gpu_options.per_process_gpu_memory_fraction = 0.3

	
	init = tf.global_variables_initializer()
	sess = tf.Session(config=config)

	if tf.gfile.Exists(FLAGS.base_dir + 'board'):
		tf.gfile.DeleteRecursively(FLAGS.base_dir + 'board')
	tf.gfile.MakeDirs(FLAGS.base_dir + 'board')


	# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
	global merged
	merged = tf.summary.merge_all()	
	global train_writer
	train_writer = tf.summary.FileWriter('./board', sess.graph)

	sess.run(init)
	coord = tf.train.Coordinator()
	tf.train.start_queue_runners(sess = sess, coord = coord)

	# vgg.load_weights('vgg16_weights.npz', sess)


def fetch_data(Train = True, Eval = True, Test = True, batch_size=None):
	global sess, train_images, train_labels, eval_images, eval_labels, test_images, test_labels
	if(batch_size != None):
		listOfarray = [None] * 6
		while(True):
			if(batch_size == 0):
				break
			else:
				_train_images, _train_labels, _eval_images, _eval_labels, _test_images, _test_labels = \
					sess.run([
						train_images, train_labels, 
						eval_images, eval_labels,
						test_images, test_labels])
				if(FLAGS.batch_size > batch_size):
					# need cutting then connect
					if(listOfarray[0] is None):
						listOfarray[0] = _train_images[:batch_size]
						listOfarray[1] = _train_labels[:batch_size]
						listOfarray[2] = _eval_images[:batch_size]
						listOfarray[3] = _eval_labels[:batch_size]
						listOfarray[4] = _test_images[:batch_size]
						listOfarray[5] = _test_labels[:batch_size]
					else:
						listOfarray[0] = np.concatenate((listOfarray[0], _train_images[:batch_size]))
						listOfarray[1] = np.concatenate((listOfarray[1], _train_labels[:batch_size]))
						listOfarray[2] = np.concatenate((listOfarray[2], _eval_images[:batch_size]))
						listOfarray[3] = np.concatenate((listOfarray[3], _eval_labels[:batch_size]))
						listOfarray[4] = np.concatenate((listOfarray[4], _test_images[:batch_size]))
						listOfarray[5] = np.concatenate((listOfarray[5], _test_labels[:batch_size]))
					break
				else:
					if(listOfarray[0] is None):
						listOfarray[0] = _train_images
						listOfarray[1] = _train_labels
						listOfarray[2] = _eval_images
						listOfarray[3] = _eval_labels
						listOfarray[4] = _test_images
						listOfarray[5] = _test_labels
					else:	
						listOfarray[0] = np.concatenate((listOfarray[0], _train_images))
						listOfarray[1] = np.concatenate((listOfarray[1], _train_labels))
						listOfarray[2] = np.concatenate((listOfarray[2], _eval_images))
						listOfarray[3] = np.concatenate((listOfarray[3], _eval_labels))
						listOfarray[4] = np.concatenate((listOfarray[4], _test_images))
						listOfarray[5] = np.concatenate((listOfarray[5], _test_labels))
					batch_size -= FLAGS.batch_size
		# tf.reshape(train_images, [batch_size, 32, 32, 3])
		# tf.reshape(eval_images, [batch_size, 32, 32, 3])
		# tf.reshape(test_images, [batch_size, 32, 32, 3])
		# tf.reshape(train_labels, [batch_size])
		# tf.reshape(eval_labels, [batch_size])
		# tf.reshape(test_labels, [batch_size])
		return listOfarray[0], listOfarray[1], listOfarray[2], listOfarray[3], listOfarray[4], listOfarray[5]
	else:
		_train_images, _train_labels, _eval_images, _eval_labels, _test_images, _test_labels = \
			sess.run([
				train_images, train_labels, 
				eval_images, eval_labels,
				test_images, test_labels]) # prefetch one batch of data
		train_X = train_Y = eval_X = eval_Y = test_X = test_Y = None

		if(Train):
			train_X = _train_images
			train_Y = _train_labels
		if(Eval):
			eval_X = _eval_images
			eval_Y = _eval_labels
		if(Test):
			test_X = _test_images
			test_Y = _test_labels
		return train_X, train_Y, eval_X, eval_Y, test_X, test_Y

def process_worker():
	global sess, fetch_data, merged, train_op, loss, global_step, images, labels, top_k_op
	strain_wk = st.STRAIN_WK(
			_worker_index = FLAGS.worker_index, 
			_check_period = FLAGS.check_period, 
			_init_base_time_step = FLAGS.base_time_step, 
			_max_steps = FLAGS.max_steps, 
			_batch_size = FLAGS.batch_size, 
			_class_num = FLAGS.class_num,
			_base_dir = FLAGS.base_dir,
			_host = FLAGS.host,
			_port_base = FLAGS.port_base,
			_s = FLAGS.s)
	# cross_entropy is only related to loss_cnt, maybe it can be deleted: huhanpeng
	build_model()
	strain_wk.register(
		_session = sess,
		_func_fetch_data = fetch_data,
		_merged = merged,
		_train_op = train_op,
		_loss = loss,
		_global_step = global_step,
		_feed_X = images,
		_feed_Y = labels,
		_top_k_op = top_k_op)

	strain_wk.run(simulate = FLAGS.sleep_time)	
	sess.close()

def process_server():
	global sess, fetch_data, merged, train_op, loss, global_step, images, labels, top_k_op
	strain_ps = st.STRAIN_PS(
		_total_worker_num = FLAGS.worker_num,
		_check_period = FLAGS.check_period, 
		_class_num = FLAGS.class_num,
		_base_dir = FLAGS.base_dir,
		_host = FLAGS.host,
		_port_base = FLAGS.port_base,
		_band_width_limit = FLAGS.band_width_limit,
		_training_end = FLAGS.training_end,
		_epsilon = FLAGS.epsilon,
		_batch_size = FLAGS.batch_size, 
		_s = FLAGS.s
		)
	build_model()
	strain_ps.register(
		_session = sess,
		_func_fetch_data = fetch_data,
		_merged = merged,
		_train_op = train_op,
		_loss = loss,
		_global_step = global_step,
		_feed_X = images,
		_feed_Y = labels,
		_top_k_op = top_k_op)
	strain_ps.run()

def main(argv):
	# preprocess
	# cifar10.maybe_download_and_extract()
	if(FLAGS.job_name == 'ps'):
		process_server()
	elif(FLAGS.job_name == 'worker'):
		process_worker()
	else:
		print("ArgumentError:argument <job_name> must be ps or worker")

''' for single machine 
import time
START_TIME = time.time()
def log(s):
	print('%f: %s' % (time.time() - START_TIME, s))
def main(argv):
	build_model()
	ep = 0
	max_steps = 1000000

	
	# train_X = np.random.rand(batch_size, height, width, 3)
	# train_Y = (np.random.rand(batch_size) * num_classes).astype(int)



	# while not self.sess.should_stop():
	while (ep < max_steps):
		time_step_start = time.time()
		ep += 1

		train_X, train_Y, eval_X, eval_Y, test_X, test_Y = fetch_data(Train = True, Eval = True, Test = True)
		

		tmp_time = time.time()
		_, cur_loss, cur_step, pred, output, train_summary = \
			sess.run([
				train_op, 
				loss, 
				global_step, 
				top_k_op, 
				logits, 
				merged], 
			feed_dict = {images: train_X, labels: train_Y})
		hete_time = time.time() - tmp_time # record the time to calulate one mini-batch only
		global train_writer
		train_writer.add_summary(train_summary, cur_step)

		# record infomation to log file
		cur_time = time.time() - START_TIME

		def return_sparcity(a):
			l = a.flatten().tolist()
			tl = [0 if _l == 0 else 1 for _l in l]
			return sum(tl) / float(len(l))
		log('Step:%d  Train loss:%.20f'	% (ep, cur_loss))
		# log('logits:{0}'.format(logits))
		log('logits:{0}'.format(output.argmax(axis=1)))
		# log('inter_logit: {0}'.format((extract_feature)))
		# log('inter_logit sparcity (# of non-zero / total) = {0}'.format(return_sparcity(extract_feature)))
		log('label: %s' %(str(train_Y)))
		log('predict result:' + str(pred+ 0))
		# log('parameters_1: %s' % (str(parameters_1)))
		print

		# record the a time	
		time_one_step = time.time() - time_step_start # record the time one step needs, excluding the sleep time 			

	log('Stop training - Step:%d\t' % (ep))
'''

if __name__ == "__main__":
  tf.app.run()


