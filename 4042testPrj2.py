import time
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


np.random.seed(10)

epochs = 1000
batch_size = 32
no_hidden1 = 30 #num of neurons in hidden layer 1
learning_rate = 0.001

# scale and normalize input data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max - X_min)
 
def normalize(X, X_mean, X_std):
    return (X - X_mean)/X_std

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

#read and divide data into test and train sets 
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

X_data, Y_data = shuffle_data(X_data, Y_data)

#separate train and test data
m = 3*X_data.shape[0] // 10
testX, testY = X_data[:m],Y_data[:m]
trainX, trainY = X_data[m:], Y_data[m:]

# scale and normalize data
trainX_max, trainX_min =  np.max(trainX, axis=0), np.min(trainX, axis=0)
trainY_max, trainY_min =  np.max(trainY, axis=0), np.min(trainY, axis=0)
testX_max, testX_min =  np.max(testX, axis=0), np.min(testX, axis=0)

trainX = scale(trainX, trainX_min, trainX_max)
testX = scale(testX, testX_min, testX_max)

trainX_mean, trainX_std = np.mean(trainX, axis=0), np.std(trainX, axis=0)
testX_mean, testX_std = np.mean(testX, axis=0), np.std(testX, axis=0)

trainX = normalize(trainX, trainX_mean, trainX_std)
testX = normalize(testX, testX_mean, testX_std)



with tf.device('/gpu:0'):
    data = np.concatenate((trainX,trainY.reshape(-1,1)),axis=1)
    trainSetNum = int(math.ceil(0.7*trainX.shape[0]))
    batches = np.array_split(data,list(np.arange(32,trainSetNum,32)))
    with tf.name_scope('input'):
        x = tf.placeholder('float64', shape=[None,8])
        y = tf.placeholder('float64', shape=[None,1])
    with tf.name_scope('constant'):
        label_max = tf.constant(trainY_max,name='maximum_output',dtype='float64')
        label_min = tf.constant(trainY_min,name='minimum_output',dtype='float64')
    with tf.name_scope('weights'):
        v = tf.Variable(tf.ones([8,30],dtype='float64'),name='h_weight')
        w = tf.Variable(tf.ones([30,1],dtype='float64'),name='o_weight')
    with tf.name_scope('biases'):   
        bh = tf.Variable(tf.zeros([1],dtype='float64'),name='h_bias')
        bo = tf.Variable(tf.zeros([1],dtype='float64'),name='o_bias')
    with tf.name_scope('h_synaptic'):
        syn_h = tf.matmul(x,v)+bh
    with tf.name_scope('h_activation'):
        act_h = tf.sigmoid(syn_h)
    with tf.name_scope('o_synaptic'):
        syn_o = tf.matmul(act_h,w)+bo
    with tf.name_scope('o_activation'):
        act_o = (label_max-label_min)*tf.sigmoid(syn_o) + label_min
    with tf.name_scope('delta'):
        delta = tf.reduce_mean(tf.square(y - act_o))
    accuracy = tf.reduce_mean(y-act_o)
    init = tf.global_variables_initializer()

learningRate = 0.001
epoch = 1000
learningerror = np.zeros(epoch)
testerror = np.zeros(epochs)
testaccuracy = np.zeros(epochs)

with tf.name_scope('train'):
    trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(delta)

with tf.Session() as sess:
    sess.run(init)
    # create log writer object
    writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph())
    for i in range(epoch):
        np.random.shuffle(batches)
        for batch in batches:
            next_train_feat = sess.run(tf.reshape((tf.stack(batch[:,:8])),[-1,8]))
            next_train_label = sess.run(tf.reshape(tf.stack(batch[:,8:]),[-1,1]))
            sess.run(trainStep,{x:next_train_feat,y:next_train_label})
        learningerror[i]=sess.run(delta,{x:sess.run(tf.reshape(tf.stack(trainX),[-1,8])),y:sess.run(tf.reshape(tf.stack(trainY),[-1,1]))})
        testerror[i],testaccuracy[i]=sess.run([delta,accuracy],{x:sess.run(tf.reshape(tf.stack(testX),[-1,8])),y:sess.run(tf.reshape(tf.stack(testY),[-1,1]))})
        print("Dataset finished[%d]: trainingErr:%lf testErr:%lf testAccuracy:%lf" % (i,learningerror[i],testerror[i],testaccuracy[i]))

#Plots
plt.figure()
plt.plot(range(epochs), train_cost, label='train error')
plt.plot(range(epochs), test_cost, label = 'test error')
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Error')
plt.title('Training and Test Errors at Alpha = %.3f'%learning_rate)
plt.legend()
plt.savefig('p_1b_sample_mse.png')
plt.show()

plt.figure()
plt.plot(range(epochs), test_accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.savefig('p_1b_sample_accuracy.png')
plt.show()