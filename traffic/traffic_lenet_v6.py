import tensorflow as tf
from tensorflow.contrib.layers import flatten
from .traffic_data import TrafficDataSets
from .data_explorer import TrainingPlotter
from .traffic_lenet import Lenet
import logging.config
logging.config.fileConfig('logging.conf')


class LenetV6Deep24x96(Lenet):

    def _LeNet(self, x, color_channel, variable_mean, variable_stddev):
        """
        new_height = (input_height - filter_height + 2 * P)/S + 1
        new_width = (input_width - filter_width + 2 * P)/S + 1
        :param x:
        :param color_channel:
        :param variable_mean:
        :param variable_stddev:
        :return:
        """
        # Hyperparameters
        mu = variable_mean
        sigma = variable_stddev

        # SOLUTION: Layer 1: Convolutional. Input = 32x32x1(or3). Output = 28x28x24.
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, color_channel, 24), mean=mu, stddev=sigma))
        conv1_b = tf.Variable(tf.zeros(24))
        conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

        # SOLUTION: Activation.
        conv1 = tf.nn.relu(conv1)

        # SOLUTION: Pooling. Input = 28x28x24. Output = 14x14x24.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # SOLUTION: Layer 2: Convolutional. Input 14x14x24 Output = 10x10x96.
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 24, 96), mean=mu, stddev=sigma))
        conv2_b = tf.Variable(tf.zeros(96))
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

        # SOLUTION: Activation.
        conv2 = tf.nn.relu(conv2)

        # SOLUTION: Pooling. Input = 10x10x96. Output = 5x5x96.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # SOLUTION: Flatten. Input = 5x5x96. Output = 2400.
        fc0 = flatten(conv2)

        # SOLUTION: Layer 3: Fully Connected. Input = 2400. Output = 400.
        fc1_W = tf.Variable(tf.truncated_normal(shape=(2400, 400), mean=mu, stddev=sigma))
        fc1_b = tf.Variable(tf.zeros(400))
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b

        # SOLUTION: Activation.
        fc1 = tf.nn.relu(fc1)

        # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_W = tf.Variable(tf.truncated_normal(shape=(400, 100), mean=mu, stddev=sigma))
        fc2_b = tf.Variable(tf.zeros(100))
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b

        # SOLUTION: Activation.
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, self.keep_prob)

        # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
        fc3_W = tf.Variable(tf.truncated_normal(shape=(100, 43), mean=mu, stddev=sigma))
        fc3_b = tf.Variable(tf.zeros(43))
        logits = tf.matmul(fc2, fc3_W) + fc3_b

        return logits
