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
	means, variances = tf.nn.moments(tf.stack(features), [0])
	normalizedFeat = data_normalization(tf.stack(features),means,variances)
	return normalizedFeat,tf.stack(labels),means,variances

def data_normalization(features,means,variances):
	normalizedFeat = (features - means) / tf.sqrt(variances)
	return normalizedFeat

#Construct trainSet and testSet according to 7:3 ratio
def split_train_test(features,labels,trainSetNum,testSetNum):
	trainLabels,testLables = tf.split(labels, [trainSetNum,testSetNum])
	trainData,testData = tf.split(features, [trainSetNum,testSetNum])
	return trainData,testData,trainLabels,testLables

#Further split to k fold
def split_k_fold(features,labels,k):
	labelFoldList = list(tf.split(labels, k))
	featFoldList= tf.split(features, k)
	return featFoldList,labelFoldList

def question1(sess,train_feat,train_labels,test_feat,test_labels,dataSize):
	# create TensorFlow Dataset objects
	tr_data = tf.contrib.data.Dataset.from_tensor_slices((train_feat, train_labels))
	test_data = tf.contrib.data.Dataset.from_tensor_slices((test_feat, test_labels))
	output_max = tf.reduce_max(tf.concat([train_labels, test_labels], 0))
	output_min = tf.reduce_min(tf.concat([train_labels, test_labels], 0))
	# create batches from dataset
	shuffled_dataset = tr_data.shuffle(dataSize)
	batched_dataset = shuffled_dataset.batch(32)
	iterator = batched_dataset.make_initializable_iterator()
	next_element = iterator.get_next()
	with tf.device('/gpu:0'):
		with tf.name_scope('input'):
			x = tf.placeholder('float64', shape=[None,8])
			y = tf.placeholder('float64', shape=[None,1])
		with tf.name_scope('constant'):
			label_max = tf.constant(sess.run(output_max),name='maximum_output')
			label_min = tf.constant(sess.run(output_min),name='minimum_output')
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
		init = tf.global_variables_initializer()


	learningRate = 0.0001
	epoch = 1000
	learningerror = np.zeros(epoch)

	with tf.name_scope('train'):
		trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(delta)

	sess.run(init)
	# create log writer object
	writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph())
	for i in range(epoch):
		sess.run(iterator.initializer)
		while True:
			try:
				next_batch = sess.run(next_element)
				next_train_feat = sess.run(tf.reshape(tf.stack(next_batch[0]),[-1,8]))
				next_train_label = sess.run(tf.reshape(tf.stack(next_batch[1]),[-1,1]))
				_,batch_delta = sess.run([trainStep,delta],{x:next_train_feat,y:next_train_label})
				learningerror[i] += batch_delta
			except tf.errors.OutOfRangeError:
				print("End of dataset")
				break;
		print("Trainingerr[%d]: %f" % (i,learningerror[i]))

	plt.figure()
	plt.plot(np.arange(epoch),learningerror)
	plt.title('Training Error')
	plt.savefig('figure_prj1.2.q1.png')
	plt.show();

features,labels,means,variances = read_files("cal_housing.data")
with tf.Session() as sess:
	trainSetNum = int(math.ceil(0.7*sess.run(tf.shape(labels))[0]))
	testSetNum = sess.run(tf.shape(labels))[0] - trainSetNum
	trainData,testData,trainLabels,testLabels = split_train_test(features,labels,trainSetNum,testSetNum)
	#question1
	question1(sess,trainData,trainLabels,testData,testLabels,trainSetNum)





		

