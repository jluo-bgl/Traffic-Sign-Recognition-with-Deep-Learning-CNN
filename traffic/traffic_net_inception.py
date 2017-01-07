import tensorflow as tf
from tensorflow.contrib.layers import flatten
from .traffic_data import TrafficDataSets
from .data_explorer import TrainingPlotter
from .traffic_lenet import Lenet
import logging.config
logging.config.fileConfig('logging.conf')


class NetInception(Lenet):

    def _LeNet(self, x, color_channel, variable_mean, variable_stddev):
        def createWeight(size, name):
            return tf.Variable(tf.truncated_normal(size, mean=variable_mean, stddev=variable_stddev),
                               name=name)
        def createBias(size, name):
            return tf.Variable(tf.constant(0.1, shape=size),
                               name=name)
        def conv2d_s1(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        def max_pool_3x3_s1(x):
            return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                                  strides=[1, 1, 1, 1], padding='SAME')

        map1 = 32
        reduce1x1 = 16
        map2 = 64
        num_fc1 = 700
        num_fc2 = 43

        # Inception Module1
        #
        # follows input
        W_conv1_1x1_1 = createWeight([1, 1, 1, map1], 'W_conv1_1x1_1')
        b_conv1_1x1_1 = createWeight([map1], 'b_conv1_1x1_1')

        # follows input
        W_conv1_1x1_2 = createWeight([1, 1, 1, reduce1x1], 'W_conv1_1x1_2')
        b_conv1_1x1_2 = createWeight([reduce1x1], 'b_conv1_1x1_2')

        # follows input
        W_conv1_1x1_3 = createWeight([1, 1, 1, reduce1x1], 'W_conv1_1x1_3')
        b_conv1_1x1_3 = createWeight([reduce1x1], 'b_conv1_1x1_3')

        # follows 1x1_2
        W_conv1_3x3 = createWeight([3, 3, reduce1x1, map1], 'W_conv1_3x3')
        b_conv1_3x3 = createWeight([map1], 'b_conv1_3x3')

        # follows 1x1_3
        W_conv1_5x5 = createWeight([5, 5, reduce1x1, map1], 'W_conv1_5x5')
        b_conv1_5x5 = createBias([map1], 'b_conv1_5x5')

        # follows max pooling
        W_conv1_1x1_4 = createWeight([1, 1, 1, map1], 'W_conv1_1x1_4')
        b_conv1_1x1_4 = createWeight([map1], 'b_conv1_1x1_4')

        # Inception Module2
        #
        # follows inception1
        W_conv2_1x1_1 = createWeight([1, 1, 4 * map1, map2], 'W_conv2_1x1_1')
        b_conv2_1x1_1 = createWeight([map2], 'b_conv2_1x1_1')

        # follows inception1
        W_conv2_1x1_2 = createWeight([1, 1, 4 * map1, reduce1x1], 'W_conv2_1x1_2')
        b_conv2_1x1_2 = createWeight([reduce1x1], 'b_conv2_1x1_2')

        # follows inception1
        W_conv2_1x1_3 = createWeight([1, 1, 4 * map1, reduce1x1], 'W_conv2_1x1_3')
        b_conv2_1x1_3 = createWeight([reduce1x1], 'b_conv2_1x1_3')

        # follows 1x1_2
        W_conv2_3x3 = createWeight([3, 3, reduce1x1, map2], 'W_conv2_3x3')
        b_conv2_3x3 = createWeight([map2], 'b_conv2_3x3')

        # follows 1x1_3
        W_conv2_5x5 = createWeight([5, 5, reduce1x1, map2], 'W_conv2_5x5')
        b_conv2_5x5 = createBias([map2], 'b_conv2_5x5')

        # follows max pooling
        W_conv2_1x1_4 = createWeight([1, 1, 4 * map1, map2], 'W_conv2_1x1_4')
        b_conv2_1x1_4 = createWeight([map2], 'b_conv2_1x1_4')

        # Fully connected layers
        # since padding is same, the feature map with there will be 4 28*28*map2
        W_fc1 = createWeight([28 * 28 * (4 * map2), num_fc1], 'W_fc1')
        b_fc1 = createBias([num_fc1], 'b_fc1')

        W_fc2 = createWeight([num_fc1, num_fc2], 'W_fc2')
        b_fc2 = createBias([num_fc2], 'b_fc2')

        # Inception Module 1
        conv1_1x1_1 = conv2d_s1(x, W_conv1_1x1_1) + b_conv1_1x1_1
        conv1_1x1_2 = tf.nn.relu(conv2d_s1(x, W_conv1_1x1_2) + b_conv1_1x1_2)
        conv1_1x1_3 = tf.nn.relu(conv2d_s1(x, W_conv1_1x1_3) + b_conv1_1x1_3)
        conv1_3x3 = conv2d_s1(conv1_1x1_2, W_conv1_3x3) + b_conv1_3x3
        conv1_5x5 = conv2d_s1(conv1_1x1_3, W_conv1_5x5) + b_conv1_5x5
        maxpool1 = max_pool_3x3_s1(x)
        conv1_1x1_4 = conv2d_s1(maxpool1, W_conv1_1x1_4) + b_conv1_1x1_4

        # concatenate all the feature maps and hit them with a relu
        inception1 = tf.nn.relu(tf.concat(3, [conv1_1x1_1, conv1_3x3, conv1_5x5, conv1_1x1_4]))

        # Inception Module 2
        conv2_1x1_1 = conv2d_s1(inception1, W_conv2_1x1_1) + b_conv2_1x1_1
        conv2_1x1_2 = tf.nn.relu(conv2d_s1(inception1, W_conv2_1x1_2) + b_conv2_1x1_2)
        conv2_1x1_3 = tf.nn.relu(conv2d_s1(inception1, W_conv2_1x1_3) + b_conv2_1x1_3)
        conv2_3x3 = conv2d_s1(conv2_1x1_2, W_conv2_3x3) + b_conv2_3x3
        conv2_5x5 = conv2d_s1(conv2_1x1_3, W_conv2_5x5) + b_conv2_5x5
        maxpool2 = max_pool_3x3_s1(inception1)
        conv2_1x1_4 = conv2d_s1(maxpool2, W_conv2_1x1_4) + b_conv2_1x1_4

        # concatenate all the feature maps and hit them with a relu
        inception2 = tf.nn.relu(tf.concat(3, [conv2_1x1_1, conv2_3x3, conv2_5x5, conv2_1x1_4]))

        # flatten features for fully connected layer
        inception2_flat = tf.reshape(inception2, [-1, 28 * 28 * 4 * map2])

        # Fully connected layers
        h_fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(inception2_flat, W_fc1) + b_fc1), self.keep_prob)

        return tf.matmul(h_fc1, W_fc2) + b_fc2
