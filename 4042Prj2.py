import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

filename_queue = tf.train.string_input_producer(["cal_housing.data", "cal_housing.domain"])

reader = tf.TextLineReader()
feature, label = reader.read(filename_queue)

features = tf.stack([tf.decode_csv(feature)])




		
