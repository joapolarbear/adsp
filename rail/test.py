import tensorflow as tf
import numpy as np
from copy import deepcopy
from returnData import returnDataA

import strain_chiller as st
flags = st.app.flags
# both worker and ps
flags.DEFINE_string('base_dir', '/home/net/hphu/usp_chiller/', 'The path where log info will be stored')
flags.DEFINE_string('host', 'localhost', 'IP address of parameter server')
flags.DEFINE_integer('port_base', 14200, 'Start port for listening to workers')
flags.DEFINE_string('job_name', '', 'Either ps or worker')
flags.DEFINE_integer('class_num', 10, 'Training batch size ')
flags.DEFINE_float('check_period', 60.0, 'Length of time between two checkpoints')
# wk
flags.DEFINE_integer('worker_index', 0, 'Index of worker')
flags.DEFINE_float('base_time_step', 20.0, 'Initial length of time between two commits for each worker')
flags.DEFINE_integer('max_steps', 1000000, 'Max steps each worker is allowed to run')
flags.DEFINE_float('sleep_time', None, 'Specify the sleep_time')
flags.DEFINE_integer('batch_size', 64, 'mini-batch size')
# ps

flags.DEFINE_integer('worker_num', 1, 'Total number of workers')
flags.DEFINE_float('band_width_limit', None, 'Specify the constrait of bandwidth in the form of x M/s')
flags.DEFINE_float('training_end', 0, 'When loss is smaller than this, end training')

FLAGS = flags.FLAGS

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

#Tool for learning model: Output X and y data for training, preidction or test
#arg:
# mode == 0: training, fixed training index: 0 to L * K on the targeted chiller
# mode == 1: predicting, fixed predict index: L * K to L on the targeted chiller
# mode == 2: all data on the targeted chiller for cross validation
# mode == 3: training 0 to L * K, clustered samples of a single chiller based on targeted days, features on targeted days (pX) required
# mode == 4: training 0 to L * K, clustered samples of multiple chillers of the SAME type based on targeted days, features on targeted days (pX) required
# mode == 5: single sample X, y; index required
# mode == 6: training 0 to L * K, all samples of multiple chillers of the SAME type based on targeted days
# mode == 7: training 0 to L * K, clustered samples of multiple chillers of the SAME type based on targeted days
def returnXy(mode=0, pX=None, index=None, K = TRAINK):
    #print '    K =', K, ' Mode =' , mode
    temp = returnDataA(1, 0)[3]
    
    #train index: 0 to L * K
    #predict index: L * K to L
    L = len(temp)

    
    if mode == 5: #single sample with index
        st = index
        ed = index + 1
    else:
        if mode == 0:
            st = 0 
            ed = int(L * K) 
        elif mode == 1:
            st = int(L * K) 
            ed = L
        elif mode == 2:
            st = 0
            ed = L
        elif mode == 3 or mode == 4 or mode == 6 or mode == 7:
            st = 0
            ed = int(L * K)
    
    #merely chiller 1
    X = []
    for xi in range(0, ed-st, 1):
        X.append([])
        
    for fi in range(TN):
        arr = returnDataA(1, fi)[3][st:ed]
        for xi in range(0, ed-st, 1):
            X[xi].append(arr[xi])
    
    y = []
    cop = []
    cop = returnDataA(1, TCOP)[3][st:ed]
    for yi in range(0, ed-st, 1):
        y.append(cop[yi])
    
    #clustered samples
    if mode == 3 or mode == 4 or mode == 6 or mode == 7: 
        X0 = []
        y0 = []
        X0 = deepcopy(X)
        y0 = deepcopy(y)
            
        #http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
        if mode == 3: #single chiller clustering            
            #print 'clustered X, y' ###
            #print X[1:5], y[1:5] ###
            #print ###
            #from sklearn.neighbors import NearestNeighbors
            neigh = NearestNeighbors()
            neigh.fit(X0) 
            
            d, xis = neigh.kneighbors([pX])
            X = []
            y = []
            for i in range(len(xis)):
                for xi in xis[i]:
                    #print xi
                    #print X0[xi]
                    X.append(X0[xi])
                    y.append(y0[xi]) #share the common index
            
        elif mode == 4: #clustering chillers of the same type
            #CN = CSN #chillers of the same type
            
            X = []
            y = []
            Xt = []
            yt = []
            for ci in range(CSN):
                
                for xi in range(0, ed-st, 1):
                    Xt.append([])
                    
                for fi in range(TN):
                    arr = returnDataA(ci, fi)[3][st:ed]
                    for xi in range(0, ed-st, 1):
                        Xt[ci*(ed-st)+xi].append(arr[xi])
                
                
                cop = []
                cop = returnDataA(ci, TCOP)[3][st:ed]
                for yi in range(0, ed-st, 1):
                    yt.append(cop[yi])
                    
            #from sklearn.neighbors import NearestNeighbors
            neigh = NearestNeighbors()#
            neigh.fit(Xt) 
            X0 = []
            y0 = []
            X0 = deepcopy(Xt)
            y0 = deepcopy(yt)
            d, xis = neigh.kneighbors([pX])
            
            for i in range(len(xis)):
                for xi in xis[i]:
                    #print xi
                    #print X0[xi]
                    X.append(X0[xi])
                    y.append(y0[xi]) #share the common index
        elif mode == 6: #clustering chillers of all types
            
            X = []
            y = []
                                    
            for ci in range(0, CAN, 1):
                for xi in range(0, ed-st, 1):
                    X.append([])
                    
                for fi in range(TN):
                    arr = returnDataA(ci, fi)[3][st:ed]
                    for xi in range(0, ed-st, 1):
                        X[(ci)*(ed-st)+xi].append(arr[xi])
                
                cop = []
                cop = returnDataA(ci, TCOP)[3][st:ed]
                for yi in range(0, ed-st, 1):
                    y.append(cop[yi])
        elif mode == 7: #clustering all chillers 
            
            X = []
            y = []
                     
            Xt = []
            yt = []
            for ci in range(CAN):
                
                for xi in range(0, ed-st, 1):
                    Xt.append([])
                    
                for fi in range(TN):
                    arr = returnDataA(ci, fi)[3][st:ed]
                    for xi in range(0, ed-st, 1):
                        Xt[ci*(ed-st)+xi].append(arr[xi])
                
                
                cop = []
                cop = returnDataA(ci, TCOP)[3][st:ed]
                for yi in range(0, ed-st, 1):
                    yt.append(cop[yi])
                    
            kmeans = KMeans(n_clusters=5)#n_clusters=2, random_state=0
            XtCs = kmeans.fit_predict(Xt) 
            X0 = []
            y0 = []
            X0 = deepcopy(Xt)
            y0 = deepcopy(yt)
            pXCs = kmeans.predict([pX])
            pXC = pXCs[0] #which cluster is pX
            
            #print X0
            #print y0
            for xi in range(len(XtCs)):
                if XtCs[xi] == pXC: #take the samples from the same cluster
                #print xi
                #print X0[xi]
                    X.append(X0[xi])
                    y.append(y0[xi]) #share the common index
            #print 'Length', len(X), len(y)
            
    return X, y



def process_worker():
	batch_size = FLAGS.batch_size
	class_num = FLAGS.class_num
	feature_size = TN

	global_step = st.train.get_or_create_global_step()
	# declare the batch_size, placeholder, variable.
	x_data = tf.placeholder(shape=[None, feature_size], dtype=tf.float32)
	y_data = tf.placeholder(shape=[None], dtype=tf.int32)
	y_target = tf.transpose(tf.one_hot(y_data, class_num))
	# y_target = tf.placeholder(shape=[class_num, None], dtype=tf.float32)
	prediction_grid = tf.placeholder(shape=[None, feature_size], dtype=tf.float32)
	b = tf.Variable(tf.random_normal(shape=[class_num, batch_size]))

	# declare the Gaussian kernel.
	gamma = tf.constant(-10.0)
	dist = tf.reduce_sum(tf.square(x_data), 1)
	dist = tf.reshape(dist, [-1, 1])
	sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
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
	loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

	# create the prediction kernel
	rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
	rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
	pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
	pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

	# create the prediction
	prediction_output = tf.matmul(tf.multiply(y_target, b), pred_kernel)
	prediction = tf.argmax(prediction_output - tf.expand_dims(tf.reduce_mean(prediction_output, 1), 1), 0)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target, 0)), tf.float32))

	# declare the optimizer function
	my_opt = tf.train.GradientDescentOptimizer(0.01)
	train_step = my_opt.minimize(loss)
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	# prepare train and test data
	tX, ty = returnXy(0)
	pX, py = returnXy(1)
	ty = [int(x/10) for x in ty]
	py = [int(x/10) for x in py]

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
		return _set = dataset(_set_x, _set_y)

	train_set = returnBatchData(tX, ty)
	test_set = returnBatchData(pX, py)

	strain_wk = st.STRAIN_WK(
            _worker_index = FLAGS.worker_index, 
            _check_period = FLAGS.check_period, 
            _init_base_time_step = FLAGS.base_time_step, 
            _max_steps = FLAGS.max_steps, 
            _batch_size = FLAGS.batch_size, 
            _class_num = class_num,
            _base_dir = FLAGS.base_dir,
            _host = FLAGS.host,
            _port_base = FLAGS.port_base)

	def fetch_data(Train = True, Eval = True, Test = True):
        train_X = train_Y = eval_X = eval_Y = test_X = test_Y = None

        if(Train):
            train_X, train_Y = train_set.next_pair()
        if(Eval):
            eval_X, eval_Y = test_set.next_pair()
        if(Test):
            test_X, test_Y = test_set.next_pair()
        return train_X, train_Y, eval_X, eval_Y, pX, py

	strain_wk.register(
			_session = sess,
			_func_fetch_data = fetch_data,
			_merged = merged,
			_train_op = my_opt,
			_loss = loss,
			_global_step = global_step,
			_feed_X = x_data,
			_feed_Y = y_data,
			_top_k_op = prediction)
	strain_wk.run(simulate = FLAGS.sleep_time)	
	sess.close()

	# Train
	loss_vec = []
	batch_accuracy = []
	for i in range(500):
		X, Y = train_set.next_pair()
		sess.run(train_step, feed_dict={x_data:X, y_data:Y})
		temp_loss = sess.run(loss, feed_dict = {x_data:X, y_data:Y})
		loss_vec.append(temp_loss)
		acc_temp = sess.run(accuracy, feed_dict = {x_data:X, y_data:Y, prediction_grid:X})
		batch_accuracy.append(acc_temp)

	print("loss:", np.mean(loss_vec), "\naccuracy", np.mean(batch_accuracy))

def process_server():
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
	strain_ps.run()

def main(argv):
	if(FLAGS.job_name == 'ps'):
		process_server()
	elif(FLAGS.job_name == 'worker'):
		process_worker()
	else:
		print("ArgumentError:argument <job_name> must be ps or worker")
	
if __name__ == "__main__":
  	st.app.run()



