import time
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


np.random.seed(10)

batch_size = 128
no_hidden1 = 30 #num of neurons in hidden layer 1


# scale and normalize input data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max - X_min)
 
def normalize(X, X_mean, X_std):
    return (X - X_mean)/X_std

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print(idx)
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
testX, testY = X_data[:m],Y_data[:m].reshape([-1,1])
trainX, trainY = X_data[m:], Y_data[m:].reshape([-1,1])

# scale and normalize data
trainX_max, trainX_min =  np.max(trainX, axis=0), np.min(trainX, axis=0)
trainX_mean, trainX_std = np.mean(trainX, axis=0), np.std(trainX, axis=0)

trainX = normalize(trainX, trainX_mean, trainX_std).astype('float32')
testX = normalize(testX, trainX_mean, trainX_std).astype('float32')
trainY = trainY.astype('float32')
testY = testY.astype('float32')
#Further split to k fold
def split(current_fold,k):
    current = current_fold
    size = int(trainX.shape[0]/k)
    
    index = np.arange(trainX.shape[0])
    lower_bound = index >= current*size
    upper_bound = index < (current + 1)*size
    cv_region = lower_bound*upper_bound

    cv_data = trainX[cv_region]
    train_data = trainX[~cv_region]
    
    cv_labels = trainY[cv_region]
    train_labels = trainY[~cv_region]
    

    return (train_data, train_labels), (cv_data, cv_labels)

def question1(train_feat,train_labels,test_feat,test_labels):
    with tf.device('/gpu:0'):
        with tf.name_scope('input'):
            x = tf.placeholder('float32', shape=[None,8])
            y = tf.placeholder('float32', shape=[None,1])
        with tf.name_scope('weights'):
            v = tf.Variable(np.random.randn(8,30)*.01,dtype='float32',name='h_weight')
            w = tf.Variable(np.random.randn(30,1)*.01,dtype='float32',name='o_weight')
        with tf.name_scope('biases'):   
            bh = tf.Variable(np.random.randn()*.01,dtype='float32',name='h_bias')
            bo = tf.Variable(np.random.randn()*.01,dtype='float32',name='o_bias')
            syn_h = tf.matmul(x,v)+bh
            act_h = tf.sigmoid(syn_h)
        with tf.name_scope('o_synaptic'):
            syn_o = tf.matmul(act_h,w)+bo
        with tf.name_scope('o_activation'):
            act_o = syn_o
        with tf.name_scope('delta'):
            delta = tf.reduce_mean(tf.square(y-act_o))
        with tf.name_scope('accu'):
            accu = tf.reduce_mean(y-act_o)
    
        init = tf.global_variables_initializer()

        learningRate = 0.0001
        epoch = 1000
        learningerror = np.zeros(epoch)
        testerror = np.zeros(epoch)
        n = len(train_feat)

        with tf.Session() as sess:
            trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(delta)
            sess.run(init) 
            # Training
            for i in range(epoch):
                _,learningerror[i] = sess.run([trainStep,delta], {x:train_feat,y:train_labels})
                test_delta = sess.run(delta, {x:test_feat, y:test_labels})
                testerror[i] = test_delta
                print("Dataset finished[%d]: %lf %lf" % (i,learningerror[i], testerror[i]))

    plt.figure()
    plt.plot(np.arange(epoch),learningerror,label='train error')
    plt.plot(np.arange(epoch), testerror,label='test error')
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Squared Error')
    plt.title('Training and Test Errors at Alpha = %.4f'%learningRate)
    plt.legend()
    plt.savefig('p_1a_sample_mse.png')
    plt.show()

def question2(train_feat,train_labels,test_feat,test_labels):
    def create_graph(idx,learningRate):
        with tf.device('/gpu:0'):
            with tf.name_scope('input'):
                x = tf.placeholder('float32', shape=[None,8])
                y = tf.placeholder('float32', shape=[None,1])
            with tf.name_scope('weights'):
                v = tf.Variable(np.random.randn(8,30)*.01,dtype='float32',name='h_weight')
                w = tf.Variable(np.random.randn(30,1)*.01,dtype='float32',name='o_weight')
            with tf.name_scope('biases'):   
                bh = tf.Variable(np.random.randn()*.01,dtype='float32',name='h_bias')
                bo = tf.Variable(np.random.randn()*.01,dtype='float32',name='o_bias')
                syn_h = tf.matmul(x,v)+bh
                act_h = tf.sigmoid(syn_h)        
            with tf.name_scope('o_synaptic'):
                syn_o = tf.matmul(act_h,w)+bo
            with tf.name_scope('o_activation'):
                act_o = syn_o
            with tf.name_scope('delta'):
                delta = tf.reduce_mean(tf.square(y-act_o))
            with tf.name_scope('accu'):
                accu = tf.reduce_mean(y-act_o)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                print('learningRate: %lf' % learningRate)
                trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(delta)
                for j in range(epoch):
                    _,error = sess.run([trainStep,delta], {x:trainX,y:trainY})
                    trainingerror[idx][j] += error/5
                    validationerror[idx][j] += sess.run(delta/5, {x:cvX, y:cvY})
    
    current_fold = 0
    learningRates=[0.001,0.0005,0.0001,0.00005,0.00001]
    learningRatesDict = {}
    epoch = 5000
    trainingerror = [np.zeros(epoch) for _ in learningRates]
    validationerror = [np.zeros(epoch) for _ in learningRates]
    # Choose the model
    for i in range(5):
        print('Current fold: {}\n'.format(current_fold + 1))
        (trainX, trainY), (cvX, cvY) = split(current_fold,5)
        for idx,learningRate in enumerate(learningRates):
            create_graph(idx,learningRate)
            tf.reset_default_graph()
        current_fold+=1
    plt.figure(1)
    plt.plot(np.arange(epoch),validationerror[0],label='alpha=0.001')
    plt.plot(np.arange(epoch),validationerror[1],label='alpha=0.0005')
    plt.plot(np.arange(epoch),validationerror[2],label='alpha=0.0001')
    plt.plot(np.arange(epoch),validationerror[3],label='alpha=0.00005')
    plt.plot(np.arange(epoch),validationerror[4],label='alpha=0.00001')
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Squared Error')
    plt.title('Validation Errors')
    plt.legend()
    plt.savefig('p_2a_model_vali_mse.png')
    plt.show()

    plt.figure(2)
    plt.plot(np.arange(epoch),trainingerror[0],label='alpha=0.001')
    plt.plot(np.arange(epoch),trainingerror[1],label='alpha=0.0005')
    plt.plot(np.arange(epoch),trainingerror[2],label='alpha=0.0001')
    plt.plot(np.arange(epoch),trainingerror[3],label='alpha=0.00005')
    plt.plot(np.arange(epoch),trainingerror[4],label='alpha=0.00001')
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Errors')
    plt.legend()
    plt.savefig('p_2b_model_train_mse.png')
    plt.show()

def question2_3(train_feat,train_labels,test_feat,test_labels):

    with tf.device('/gpu:0'):
        with tf.name_scope('input'):
            x = tf.placeholder('float32', shape=[None,8])
            y = tf.placeholder('float32', shape=[None,1])
        with tf.name_scope('weights'):
            v = tf.Variable(np.random.randn(8,30)*.01,dtype='float32',name='h_weight')
            w = tf.Variable(np.random.randn(30,1)*.01,dtype='float32',name='o_weight')
        with tf.name_scope('biases'):   
            bh = tf.Variable(np.random.randn()*.01,dtype='float32',name='h_bias')
            bo = tf.Variable(np.random.randn()*.01,dtype='float32',name='o_bias')
            syn_h = tf.matmul(x,v)+bh
            act_h = tf.sigmoid(syn_h)
        with tf.name_scope('o_synaptic'):
            syn_o = tf.matmul(act_h,w)+bo
        with tf.name_scope('o_activation'):
            act_o = syn_o
        with tf.name_scope('delta'):
            delta = tf.reduce_mean(tf.square(y-act_o))
    
    init = tf.global_variables_initializer()
    # Optimal learningRate
    learningRate = 0.001
    epoch = 1000
    testerror = np.zeros(epoch)
    n = len(train_feat)

    with tf.Session() as sess:
        trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(delta)
        sess.run(init) 
        # Training
        for i in range(epoch):
            sess.run(trainStep, {x:train_feat,y:train_labels})
            test_delta = sess.run(delta, {x:test_feat, y:test_labels})
            testerror[i] = test_delta
            print("Dataset finished[%d]: %lf" % (i,testerror[i]))


    plt.figure()
    plt.plot(np.arange(epoch), testerror)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Square Error')
    plt.title('Test Error Under Optimal Learning Rate')
    plt.savefig('p_2b_optimalLearn_accuracy.png')
    plt.show()

def question3(train_feat,train_labels,test_feat,test_labels):
    def create_graph(idx,neuronNum,learningRate):
        with tf.device('/gpu:0'):
            with tf.name_scope('input'):
                x = tf.placeholder('float32', shape=[None,8])
                y = tf.placeholder('float32', shape=[None,1])
            with tf.name_scope('weights'):
                v = tf.Variable(np.random.randn(8,neuronNum)*.01,dtype='float32',name='h_weight')
                w = tf.Variable(np.random.randn(neuronNum,1)*.01,dtype='float32',name='o_weight')
            with tf.name_scope('biases'):   
                bh = tf.Variable(np.random.randn()*.01,dtype='float32',name='h_bias')
                bo = tf.Variable(np.random.randn()*.01,dtype='float32',name='o_bias')
                syn_h = tf.matmul(x,v)+bh
                act_h = tf.sigmoid(syn_h)        
            with tf.name_scope('o_synaptic'):
                syn_o = tf.matmul(act_h,w)+bo
            with tf.name_scope('o_activation'):
                act_o = syn_o
            with tf.name_scope('delta'):
                delta = tf.reduce_mean(tf.square(y-act_o))
            with tf.name_scope('accu'):
                accu = tf.reduce_mean(y-act_o)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                print('NeuronNum: %d' % neuronNum)
                trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(delta)
                for j in range(epoch):
                    _,error = sess.run([trainStep,delta], {x:trainX,y:trainY})
                    trainingerror[idx][j] += error/5
                    validationerror[idx][j] += sess.run(delta/5, {x:cvX, y:cvY})

    current_fold = 0
    learningRate = 0.001
    hidden_neuron_nums = [20,30,40,50,60]
    epoch = 1000
    trainingerror = [np.zeros(epoch) for _ in hidden_neuron_nums]
    validationerror = [np.zeros(epoch) for _ in hidden_neuron_nums]

    for i in range(5):
        print('Current fold: {}\n'.format(current_fold + 1))
        (trainX, trainY), (cvX, cvY) = split(current_fold,5)           
        for idx,neuronNum in enumerate(hidden_neuron_nums):
            create_graph(idx,neuronNum,learningRate)
            tf.reset_default_graph()
        current_fold+=1

    plt.figure(1)
    for idx,num in enumerate(hidden_neuron_nums):
        plt.plot(np.arange(epoch),validationerror[idx],label='hiddenNeuronNum='+str(num))
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Squared Error')
    plt.title('Validation Errors For Different Hidden Neurons')
    plt.legend()
    plt.savefig('p_3b_model_vali_mse.png')
    plt.show()

    plt.figure(2)
    for idx,num in enumerate(hidden_neuron_nums):
        plt.plot(np.arange(epoch),trainingerror[idx],label='hiddenNeuronNum='+str(num))
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Errors For Different Hidden Neurons')
    plt.legend()
    plt.savefig('p_3a_model_train_mse.png')
    plt.show()

def question3_2(train_feat,train_labels,test_feat,test_labels):

    with tf.device('/gpu:0'):
        with tf.name_scope('input'):
            x = tf.placeholder('float32', shape=[None,8])
            y = tf.placeholder('float32', shape=[None,1])
        with tf.name_scope('weights'):
            v = tf.Variable(np.random.randn(8,60)*.01,dtype='float32',name='h_weight')
            w = tf.Variable(np.random.randn(60,1)*.01,dtype='float32',name='o_weight')
        with tf.name_scope('biases'):   
            bh = tf.Variable(np.random.randn()*.01,dtype='float32',name='h_bias')
            bo = tf.Variable(np.random.randn()*.01,dtype='float32',name='o_bias')
            syn_h = tf.matmul(x,v)+bh
            act_h = tf.sigmoid(syn_h)
        with tf.name_scope('o_synaptic'):
            syn_o = tf.matmul(act_h,w)+bo
        with tf.name_scope('o_activation'):
            act_o = syn_o
        with tf.name_scope('delta'):
            delta = tf.reduce_mean(tf.square(y-act_o))
    
    init = tf.global_variables_initializer()
    # Optimal learningRate
    learningRate = 0.001
    epoch = 1000
    testerror = np.zeros(epoch)
    n = len(train_feat)

    with tf.Session() as sess:
        trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(delta)
        sess.run(init) 
        # Training
        for i in range(epoch):
            sess.run(trainStep, {x:train_feat,y:train_labels})
            test_delta = sess.run(delta, {x:test_feat, y:test_labels})
            testerror[i] = test_delta
            print("Dataset finished[%d]: %lf" % (i,testerror[i]))


    plt.figure()
    plt.plot(np.arange(epoch), testerror)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Square Error')
    plt.title('Test Error With Optimal Neuron Number')
    plt.savefig('p_3b_optimalNeuronNum_test_error.png')
    plt.show()

def question4(train_feat,train_labels,test_feat,test_labels):
    g1 = tf.Graph()
    g2 = tf.Graph()
    g3 = tf.Graph()

    # Optimal learningRate
    learningRate = 0.001
    epoch = 1000
    testerror = [np.zeros(epoch) for _ in range(3)]
    n = len(train_feat)

    with g1.as_default():
        sess1 = tf.Session(graph=g1)
        with tf.name_scope('input'):
            x = tf.placeholder('float32', shape=[None,8])
            y = tf.placeholder('float32', shape=[None,1])
        with tf.name_scope('weights'):
            v = tf.Variable(np.random.randn(8,60)*.01,dtype='float32',name='h_weight')
            w = tf.Variable(np.random.randn(60,1)*.01,dtype='float32',name='o_weight')
        with tf.name_scope('biases'):   
            bh = tf.Variable(np.random.randn()*.01,dtype='float32',name='h_bias')
            bo = tf.Variable(np.random.randn()*.01,dtype='float32',name='o_bias')
            syn_h = tf.matmul(x,v)+bh
            act_h = tf.sigmoid(syn_h)
        with tf.name_scope('o_synaptic'):
            syn_o = tf.matmul(act_h,w)+bo
        with tf.name_scope('o_activation'):
            act_o = syn_o
        with tf.name_scope('delta'):
            delta = tf.reduce_mean(tf.square(y-act_o))
        trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(delta)
        sess1.run(tf.global_variables_initializer())
        print('Current Layer: 3') 
        # Training
        for i in range(epoch):
            sess1.run(trainStep, {x:train_feat,y:train_labels})
            test_delta = sess1.run(delta, {x:test_feat, y:test_labels})
            testerror[0][i] = test_delta

    with g2.as_default():
        sess2 = tf.Session(graph=g2)
        with tf.name_scope('input'):
            x = tf.placeholder('float32', shape=[None,8])
            y = tf.placeholder('float32', shape=[None,1])
        with tf.name_scope('weights'):
            v = tf.Variable(np.random.randn(8,60)*.01,dtype='float32',name='h_weight')
            v1 = tf.Variable(np.random.randn(60,60)*.01,dtype='float32',name='h1_weight')
            w = tf.Variable(np.random.randn(60,1)*.01,dtype='float32',name='o_weight')
        with tf.name_scope('biases'):   
            bh = tf.Variable(np.random.randn()*.01,dtype='float32',name='h_bias')
            bh1 = tf.Variable(np.random.randn()*.01,dtype='float32',name='h1_bias')
            bo = tf.Variable(np.random.randn()*.01,dtype='float32',name='o_bias')
            syn_h = tf.matmul(x,v)+bh
            act_h = tf.sigmoid(syn_h)
            syn_h1 = tf.matmul(act_h,v1)+bh1
            act_h1 = tf.sigmoid(syn_h1)
        with tf.name_scope('o_synaptic'):
            syn_o = tf.matmul(act_h1,w)+bo
        with tf.name_scope('o_activation'):
            act_o = syn_o
        with tf.name_scope('delta'):
            delta = tf.reduce_mean(tf.square(y-act_o))
        trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(delta)
        sess2.run(tf.global_variables_initializer())
        print('Current Layer: 4') 
        # Training
        for i in range(epoch):
            sess2.run(trainStep, {x:train_feat,y:train_labels})
            test_delta = sess2.run(delta, {x:test_feat, y:test_labels})
            testerror[1][i] = test_delta

    with g3.as_default():
        sess3 = tf.Session(graph=g3)
        with tf.name_scope('input'):
            x = tf.placeholder('float32', shape=[None,8])
            y = tf.placeholder('float32', shape=[None,1])
        with tf.name_scope('weights'):
            v = tf.Variable(np.random.randn(8,60)*.01,dtype='float32',name='h_weight')
            v1 = tf.Variable(np.random.randn(60,60)*.01,dtype='float32',name='h1_weight')
            v2 = tf.Variable(np.random.randn(60,60)*.01,dtype='float32',name='h2_weight')
            w = tf.Variable(np.random.randn(60,1)*.01,dtype='float32',name='o_weight')
        with tf.name_scope('biases'):   
            bh = tf.Variable(np.random.randn()*.01,dtype='float32',name='h_bias')
            bh1 = tf.Variable(np.random.randn()*.01,dtype='float32',name='h1_bias')
            bh2 = tf.Variable(np.random.randn()*.01,dtype='float32',name='h2_bias')
            bo = tf.Variable(np.random.randn()*.01,dtype='float32',name='o_bias')
            syn_h = tf.matmul(x,v)+bh
            act_h = tf.sigmoid(syn_h)
            syn_h1 = tf.matmul(act_h,v1)+bh1
            act_h1 = tf.sigmoid(syn_h1)
            syn_h2 = tf.matmul(act_h1,v2)+bh1
            act_h2 = tf.sigmoid(syn_h2)
        with tf.name_scope('o_synaptic'):
            syn_o = tf.matmul(act_h2,w)+bo
        with tf.name_scope('o_activation'):
            act_o = syn_o
        with tf.name_scope('delta'):
            delta = tf.reduce_mean(tf.square(y-act_o))
        trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(delta)
        sess3.run(tf.global_variables_initializer())
        print('Current Layer: 5') 
        # Training
        for i in range(epoch):
            sess3.run(trainStep, {x:train_feat,y:train_labels})
            test_delta = sess3.run(delta, {x:test_feat, y:test_labels})
            testerror[2][i] = test_delta


    plt.figure()
    plt.plot(np.arange(epoch), testerror[0],label='hiddenLayerNum=3')
    plt.plot(np.arange(epoch), testerror[1],label='hiddenLayerNum=4')
    plt.plot(np.arange(epoch), testerror[2],label='hiddenLayerNum=5')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Test Error For Different Hidden Layer Numbers')
    plt.legend()
    plt.savefig('p_4_optimalLayer_test_error.png')
    #plt.show()


question1(trainX,trainY,testX,testY)
#question2(trainX,trainY,testX,testY)
#question2_3(trainX,trainY,testX,testY)
#question3(trainX,trainY,testX,testY)
#question3_2(trainX,trainY,testX,testY)
#question4(trainX,trainY,testX,testY)