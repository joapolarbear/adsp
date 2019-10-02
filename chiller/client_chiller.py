import numpy as np
import math
from copy import deepcopy
from returnData import returnDataA

import strain_chiller as st
import tensorflow as tf
flags = tf.app.flags
# both worker and ps
flags.DEFINE_string('base_dir', './', 'The path where log info will be stored')
flags.DEFINE_string('host', 'localhost', 'IP address of parameter server')
flags.DEFINE_integer('port_base', 14300, 'Start port for listening to workers')
flags.DEFINE_string('job_name', '', 'Either ps or worker')
flags.DEFINE_integer('class_num', 10, 'Training batch size ')
flags.DEFINE_float('check_period', 60.0, 'Length of time between two checkpoints')
# wk
flags.DEFINE_integer('worker_index', 0, 'Index of worker')
flags.DEFINE_float('base_time_step', 20.0, 'Initial length of time between two commits for each worker')
flags.DEFINE_integer('max_steps', 1000000, 'Max steps each worker is allowed to run')
flags.DEFINE_float('sleep_time', 0, 'Specify the sleep_time')
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

sess = None
fetch_data = None
merged = None
train_op = None
loss = None
global_step = None
x_data = None
y_data = None
prediction = None
prediction_grid = None

train_set = None
test_set = None

train_X = None
train_Y = None
eval_X = None
eval_Y = None
test_X = None
test_Y = None


def build_model():
    global sess, fetch_data, merged, train_op, loss, global_step, x_data, y_data, prediction, prediction_grid
    global TN
    feature_size = TN
    class_num = FLAGS.class_num
    batch_size = FLAGS.batch_size
    global_step = tf.train.get_or_create_global_step()
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
    tf.summary.scalar('loss', loss)

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
    my_opt = tf.train.GradientDescentOptimizer(0.001)
    train_op = my_opt.minimize(loss=loss, global_step=global_step)
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all() 
    sess = tf.Session()
    sess.run(init)
    
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
    batch_cnt = len(x) / (FLAGS.batch_size * TN)
    input_len = batch_cnt * (FLAGS.batch_size * TN)
    _set_x = np.asarray(shuffle_tx[:input_len]).reshape(-1, FLAGS.batch_size, TN)
    _set_y = np.asarray(shuffle_ty[:input_len]).reshape(-1, FLAGS.batch_size)
    # return set
    return dataset(_set_x, _set_y)

def prepare_data():
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

    def check_classify(y_):
        cnt = [0 for _ in xrange(FLAGS.class_num)]
        for yi in y_:
          if(yi >= 0 and yi <= FLAGS.class_num - 1):
              cnt[yi] += 1
        print
        print("y_: %s\n total num: %d, sum of cnt %d" %(str(cnt), len(y_), sum(cnt)))
    check_classify(ally)

    train_indices = np.random.choice(len(allx), int(round(len(allx)*TRAINK)), replace=False)
    test_indices = np.array(list(set(range(len(allx))) - set(train_indices)))
    tx = np.array(allx)[train_indices]
    ty = np.array(ally)[train_indices]
    px = np.array(allx)[test_indices]
    py = np.array(ally)[test_indices]
    
    global train_set, test_set
    train_set = returnBatchData(tx, ty)
    test_set = returnBatchData(px, py)

def fetch_data(Train = True, Eval = True, Test = True):
    global train_set, test_set
    global train_X, train_Y, eval_X, eval_Y, test_X, test_Y 
    train_X = train_Y = eval_X = eval_Y = test_X = test_Y = None

    if(Train):
        train_X, train_Y = train_set.next_pair()
    if(Eval):
        eval_X, eval_Y = test_set.next_pair()
    if(Test):
        test_X, test_Y = test_set.next_pair()
    return train_X, train_Y, eval_X, eval_Y, test_X, test_Y

def process_worker():
    global sess, fetch_data, merged, train_op, loss, global_step, x_data, y_data, prediction_grid, prediction
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

    build_model()
    prepare_data()

    strain_wk.register(
            _session = sess,
            _func_fetch_data = fetch_data,
            _merged = merged,
            _train_op = train_op,
            _loss = loss,
            _global_step = global_step,
            _feed_X = x_data,
            _feed_Y = y_data,
            _feed_prediction = prediction_grid,
            _top_k_op = prediction)
    strain_wk.run(simulate = FLAGS.sleep_time)  
    sess.close()


def process_server():
    global sess, fetch_data, merged, train_op, loss, global_step, x_data, y_data, prediction_grid, prediction
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
    prepare_data()

    strain_ps.register(
        _session = sess, 
        _func_fetch_data = fetch_data,
        _merged = merged,
        _train_op = train_op,
        _loss = loss,
        _global_step = global_step,
        _feed_X = x_data,
        _feed_Y = y_data,
        _feed_prediction = prediction_grid,
        _top_k_op = prediction
        )
    strain_ps.run()
    sess.close()

def main(argv):
    if(FLAGS.job_name == 'ps'):
        process_server()
    elif(FLAGS.job_name == 'worker'):
        process_worker()
    else:
        print("ArgumentError:argument <job_name> must be ps or worker")
    
if __name__ == "__main__":
    st.app.run()



