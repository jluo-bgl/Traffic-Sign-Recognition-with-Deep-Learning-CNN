import tensorflow as tf
from tensorflow.contrib.layers import flatten
from .traffic_data import TrafficDataSets
from .data_explorer import TrainingPlotter
from .traffic_lenet import Lenet
import logging.config
logging.config.fileConfig('logging.conf')

class LenetV7LessMaxPooling(Lenet):

    # LeNet architecture:
    # INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
    # create the LeNet and return the result of the last fully connected layer.
    # new_height = (input_height - filter_height + 2 * P) / S + 1
    # new_width = (input_width - filter_width + 2 * P) / S + 1
    def _LeNet(self, x, color_channel, variable_mean, variable_stddev):
        # Hyperparameters
        mu = variable_mean
        sigma = variable_stddev

        # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, color_channel, 6), mean=mu, stddev=sigma))
        conv1_b = tf.Variable(tf.zeros(6))
        conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

        # SOLUTION: Activation.
        conv1 = tf.nn.relu(conv1)

        # SOLUTION: Pooling. Input = 28x28x6. Output = 27x27x6.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

        # SOLUTION: Layer 2: Convolutional. Output = 23x23x16.
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
        conv2_b = tf.Variable(tf.zeros(16))
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

        # SOLUTION: Activation.
        conv2 = tf.nn.relu(conv2)

        # SOLUTION: Pooling. Input = 23x23x16. Output = 12x12x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # SOLUTION: Flatten. Input = 12x12x16. Output = 2304.
        fc0 = flatten(conv2)

        # SOLUTION: Layer 3: Fully Connected. Input = 2304. Output = 512.
        fc1_W = tf.Variable(tf.truncated_normal(shape=(2304, 512), mean=mu, stddev=sigma))
        fc1_b = tf.Variable(tf.zeros(512))
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b

        # SOLUTION: Activation.
        fc1 = tf.nn.relu(fc1)

        # SOLUTION: Layer 4: Fully Connected. Input = 512. Output = 84.
        fc2_W = tf.Variable(tf.truncated_normal(shape=(512, 130), mean=mu, stddev=sigma))
        fc2_b = tf.Variable(tf.zeros(130))
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b

        # SOLUTION: Activation.
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, self.keep_prob)

        # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
        fc3_W = tf.Variable(tf.truncated_normal(shape=(130, 43), mean=mu, stddev=sigma))
        fc3_b = tf.Variable(tf.zeros(43))
        logits = tf.matmul(fc2, fc3_W) + fc3_b

        return logits
