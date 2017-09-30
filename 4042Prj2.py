import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#Read csv file and convert it to tf tensor with random shuffle done
def read_files(filename):
	featUseCols = (0,1,2,3,4,5,6,7)
	labelUseCols = (8)
	colNames = "feat0,feat1,feat2,feat3,feat4,feat5,feat6,feat7,label"
	features = np.genfromtxt(filename, delimiter=',',autostrip=True, usecols=featUseCols)
	labels = np.genfromtxt(filename, delimiter=',',autostrip=True, usecols=labelUseCols)

	#Perform data normalization and feature scaling
	means = np.mean(features,0)
	std = np.mean(features)
	normalizedFeat = data_normalization(features,means,std)
	return normalizedFeat,labels,means,std

def data_normalization(features,means,std):
	normalizedFeat = (features - means) / std
	return normalizedFeat

#Construct trainSet and testSet according to 7:3 ratio
def split_train_test(features,labels,trainSetNum):
	indices = np.random.permutation(features.shape[0])
	training_idx, test_idx = indices[:trainSetNum], indices[trainSetNum:]
	trainLabels,testLables = labels[training_idx], labels[test_idx]
	trainData,testData = features[training_idx,:], features[test_idx,:]
	return trainData,testData,trainLabels,testLables

#Further split to k fold
"""def split_k_fold(features,labels,k):
	labelFoldList = list(tf.split(labels, k))
	featFoldList= tf.split(features, k)
	return featFoldList,labelFoldList"""

def question1(train_feat,train_labels,test_feat,test_labels,dataSize):
	data = np.concatenate((train_feat,train_labels.reshape(-1,1)),axis=1)
	batches = np.array_split(data,list(np.arange(32,dataSize,32)))
	labels = np.concatenate((train_labels,test_labels),axis=0)
	output_max = np.amax(labels)
	output_min = np.amin(labels)
	print (output_max,output_min)
	
	with tf.name_scope('input'):
		x = tf.placeholder('float64', shape=[None,8])
		y = tf.placeholder('float64', shape=[None,1])
	with tf.name_scope('weights'):
		v = tf.Variable(np.random.randn(8,30)*.01,dtype='float64',name='h_weight')
		#v1 = tf.Variable(tf.truncated_normal([30,20],dtype='float64'),name='h_weight')
		#v2 = tf.Variable(tf.truncated_normal([20,20],dtype='float64'),name='h_weight')
		w = tf.Variable(np.random.randn(30,1)*.01,dtype='float64',name='o_weight')
	with tf.name_scope('biases'):	
		bh = tf.Variable(tf.zeros([1],dtype='float64'),name='h_bias')
		#bh1 = tf.Variable(tf.zeros([1],dtype='float64'),name='h_bias')
		#bh2 = tf.Variable(tf.zeros([1],dtype='float64'),name='h_bias')
		bo = tf.Variable(tf.zeros([1],dtype='float64'),name='o_bias')
		syn_h = tf.matmul(x,v)+bh
		act_h = tf.sigmoid(syn_h)
		#syn_h1 = tf.matmul(act_h,v1)+bh1
		#act_h1 = tf.sigmoid(syn_h1)
		#syn_h2 = tf.matmul(act_h1,v2)+bh2
		#act_h2 = tf.sigmoid(syn_h2)
	with tf.name_scope('o_synaptic'):
		syn_o = tf.matmul(act_h,w)+bo
	with tf.name_scope('o_activation'):
		#act_o = (output_max-output_min)*tf.sigmoid(syn_o) + output_min
		act_o = syn_o
	with tf.name_scope('delta'):
		delta = tf.reduce_mean(tf.square(y-act_o))
		accu = tf.reduce_mean(act_o-y)
	init = tf.global_variables_initializer()


	learningRate = 0.001
	epoch = 1000
	learningerror = np.zeros(epoch)
	trainaccuracy = np.zeros(epoch)
	testaccuracy = np.zeros(epoch)

	with tf.name_scope('train'):
		trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(delta)
	with tf.Session() as sess:
		sess.run(init) 
		# create log writer object
		for i in range(epoch):
			np.random.shuffle(data)
			batches = np.array_split(data,list(np.arange(32,dataSize,32)))
			for batch in batches:
				_,batch_delta = sess.run([trainStep,delta], {x:batch[:,:8], y:batch[:,8:]})
			_,train_delta,train_accu = sess.run([trainStep,delta,accu], {x:trainData[:,:8], y:trainLabels[:].reshape(-1,1)})
			_,test_delta,test_accu = sess.run([trainStep,delta,accu], {x:testData[:,:8], y:testLabels[:].reshape(-1,1)})
			learningerror[i] = train_delta
			trainaccuracy[i] = train_accu
			testaccuracy[i] = test_accu
			print("Dataset finished[%d]: %lf %lf %lf" % (i,learningerror[i], trainaccuracy[i], testaccuracy[i]))

	plt.figure()
	plt.plot(np.arange(epoch),learningerror)
	plt.title('Training Error')
	plt.savefig('figure_prj1.2.q1.png')
	plt.show();

	plt.figure()
	plt.plot(np.arange(epoch), testaccuracy)
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title('Test Accuracy')
	plt.savefig('p_1b_sample_accuracy.png')
	plt.show()

features,labels,means,std = read_files("cal_housing.data")
trainSetNum = int(math.ceil(0.7*labels.shape[0]))
testSetNum = labels.shape[0] - trainSetNum
print(features.shape,labels.shape)
trainData, testData, trainLabels, testLabels = split_train_test(features,labels,trainSetNum)
#question1
question1(trainData,trainLabels,testData,testLabels,trainSetNum)





		


