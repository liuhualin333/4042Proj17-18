import tensorflow as tf
import numpy as np
import math

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
	means_label = np.mean(labels,0)
	std_label = np.mean(labels)
	normalizedFeat = data_normalization(features,means,std)
	normalizedLabels = data_normalization(labels,means_label,std_label)
	return normalizedFeat,normalizedLabels,means,std,means_label,std_label

def data_normalization(features,means,std):
	normalizedFeat = (features - means) / std
	return normalizedFeat

#Construct trainSet and testSet according to 7:3 ratio
def split_train_test(features,labels,trainSetNum):
	indices = np.random.permutation(features.shape[0])
	training_idx, test_idx = indices[:trainSetNum], indices[trainSetNum:]
	trainLabels,testLables = labels[training_idx], labels[test_idx]
	trainData,testData = features[training_idx,:], features[test_idx,:]
	print(trainData,trainLabels)
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
	
	with tf.name_scope('input'):
		x = tf.placeholder('float64', shape=[None,8])
		y = tf.placeholder('float64', shape=[None,1])
	with tf.name_scope('constant'):
		label_max = tf.constant(output_max,name='maximum_output',dtype='float64')
		label_min = tf.constant(output_min,name='minimum_output',dtype='float64')
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
		#act_o = (label_max-label_min)*tf.sigmoid(syn_o) + label_min
		act_o = syn_o
	with tf.name_scope('delta'):
		delta = tf.reduce_mean(tf.square(y - act_o))
	init = tf.global_variables_initializer()


	learningRate = 0.001
	epoch = 1000
	learningerror = np.zeros(epoch)

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
			learningerror[i]=sess.run(delta,{x:sess.run(tf.reshape(tf.stack(train_feat),[-1,8])),y:sess.run(tf.reshape(tf.stack(train_labels),[-1,1]))})
			print("Dataset finished[%d]: %lf" % (i,learningerror[i]))

	plt.figure()
	plt.plot(np.arange(epoch),learningerror)
	plt.title('Training Error')
	plt.savefig('figure_prj1.2.q1.png')
	plt.show();

features,labels,means,std,means_label,std_label = read_files("cal_housing.data")
trainSetNum = int(math.ceil(0.7*labels.shape[0]))
testSetNum = labels.shape[0] - trainSetNum
print(features.shape,labels.shape)
trainData,testData,trainLabels,testLabels = split_train_test(features,labels,trainSetNum)
#question1
question1(trainData,trainLabels,testData,testLabels,trainSetNum)





		


