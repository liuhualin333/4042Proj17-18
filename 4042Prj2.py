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

	#Random permutation of data
	idx = np.random.permutation(len(features))

	#Perform feature scaling with zero mean and gaussian distribution
	means, variances = tf.nn.moments(tf.stack(features[idx]), [0])
	normalizedFeat = (tf.stack(features[idx]) - means) / tf.sqrt(variances)
	
	return normalizedFeat,tf.stack(labels[idx]),means,variances

#Construct trainSet and testSet according to 7:3 ratio
def split_train_test(features,labels,trainSetNum,testSetNum):
	trainLabels,testLables = tf.split(labels, [trainSetNum,testSetNum])
	trainData,testData = tf.split(features, [trainSetNum,testSetNum])
	return trainData,testData,trainLabels,testLables

features,labels,means,variances = read_files("cal_housing.data")

with tf.Session() as sess:
	trainSetNum = int(math.ceil(0.7*sess.run(tf.shape(labels))[0]))
	testSetNum = sess.run(tf.shape(labels))[0] - trainSetNum
	trainData,testData,trainLabels,testLabels = split_train_test(features,labels,trainSetNum,testSetNum)






		
