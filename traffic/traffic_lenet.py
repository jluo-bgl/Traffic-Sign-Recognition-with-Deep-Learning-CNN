import unittest
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from .traffic_data import TrafficDataSets
import logging.config
logging.config.fileConfig('logging.conf')

class Lenet(object):

    def __init__(self):
        self.EPOCHS = 10
        self.BATCH_SIZE = 50
        self.LABEL_SIZE = TrafficDataSets.NUMBER_OF_CLASSES

        self.traffic_datas = TrafficDataSets('train.p', 'test.p')

        # MNIST consists of 28x28x1, grayscale images
        self.x = tf.placeholder(tf.float32, (None, 32, 32, 3))
        # Classify over 10 digits 0-9
        self.y = tf.placeholder(tf.float32, (None, self.LABEL_SIZE))
        self.fc2 = Lenet._LeNet(self, self.x)

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.fc2, self.y))
        self.opt = tf.train.AdamOptimizer()
        self.train_op = self.opt.minimize(self.loss_op)
        self.correct_prediction = tf.equal(tf.argmax(self.fc2, 1), tf.argmax(self.y, 1))
        self.accuracy_op = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    # LeNet architecture:
    # INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
    # create the LeNet and return the result of the last fully connected layer.
    def _LeNet(self, x):
        # x is 32, 32, 3
        # Reshape from 2D to 4D. This prepares the data for
        # convolutional and pooling layers.
        x = tf.reshape(x, (-1, 32, 32, 3))
        # Pad 0s to 32x32. Centers the digit further.
        # Add 2 rows/columns on each side for height and width dimensions.
        # x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")

        # 28x28x6
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6)))
        conv1_b = tf.Variable(tf.zeros(6))
        conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

        conv1 = tf.nn.relu(conv1)

        # 14x14x6
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 10x10x16
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16)))
        conv2_b = tf.Variable(tf.zeros(16))
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

        conv2 = tf.nn.relu(conv2)

        # 5x5x16
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Flatten
        fc1 = flatten(conv2)
        # (5 * 5 * 16, 120)
        fc1_shape = (fc1.get_shape().as_list()[-1], 120)

        fc1_W = tf.Variable(tf.truncated_normal(shape=(fc1_shape)))
        fc1_b = tf.Variable(tf.zeros(120))
        fc1 = tf.matmul(fc1, fc1_W) + fc1_b
        fc1 = tf.nn.relu(fc1)

        fc2_W = tf.Variable(tf.truncated_normal(shape=(120, self.LABEL_SIZE)))
        fc2_b = tf.Variable(tf.zeros(self.LABEL_SIZE))
        return tf.matmul(fc1, fc2_W) + fc2_b

    def eval_data(self, dataset):
        """
        Given a dataset as input returns the loss and accuracy.
        """
        # If dataset.num_examples is not divisible by BATCH_SIZE
        # the remainder will be discarded.
        # Ex: If BATCH_SIZE is 64 and training set has 55000 examples
        # steps_per_epoch = 55000 // 64 = 859
        # num_examples = 859 * 64 = 54976
        #
        # So in that case we go over 54976 examples instead of 55000.
        steps_per_epoch = dataset.num_examples // self.BATCH_SIZE
        num_examples = steps_per_epoch * self.BATCH_SIZE
        total_acc, total_loss = 0, 0
        sess = tf.get_default_session()
        for step in range(steps_per_epoch):
            batch_x, batch_y = dataset.next_batch(self.BATCH_SIZE)
            loss, acc = sess.run([self.loss_op, self.accuracy_op], feed_dict={self.x: batch_x, self.y: batch_y})
            total_acc += (acc * batch_x.shape[0])
            total_loss += (loss * batch_x.shape[0])
        return total_loss / num_examples, total_acc / num_examples

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            steps_per_epoch = self.traffic_datas.train.num_examples // self.BATCH_SIZE

            # Train model
            for i in range(self.EPOCHS):
                for step in range(steps_per_epoch):
                    batch_x, batch_y = self.traffic_datas.train.next_batch(self.BATCH_SIZE)
                    loss = sess.run(self.train_op, feed_dict={self.x: batch_x, self.y: batch_y})

                val_loss, val_acc = self.eval_data(self.traffic_datas.validation)
                logging.info("EPOCH {} ...".format(i + 1))
                logging.info("Validation loss = {:.3f}".format(val_loss))
                logging.info("Validation accuracy = {:.3f}".format(val_acc))

            # Evaluate on the test data
            test_loss, test_acc = self.eval_data(self.traffic_datas.test)
            logging.info("Test loss = {:.3f}".format(test_loss))
            logging.info("Test accuracy = {:.3f}".format(test_acc))


class TestLenet(unittest.TestCase):
    def test_lenet(self):
        Lenet().train()


"""
EPOCH 50, Batch=500
2016-12-21 23:26:04,443 - EPOCH 1 ...
2016-12-21 23:26:04,443 - Validation loss = 2091.946
2016-12-21 23:26:04,443 - Validation accuracy = 0.062
2016-12-21 23:26:32,802 - EPOCH 2 ...
2016-12-21 23:26:32,803 - Validation loss = 1526.941
2016-12-21 23:26:32,803 - Validation accuracy = 0.067
2016-12-21 23:27:05,281 - EPOCH 3 ...
2016-12-21 23:27:05,281 - Validation loss = 1205.399
2016-12-21 23:27:05,281 - Validation accuracy = 0.080
2016-12-21 23:27:33,481 - EPOCH 4 ...
2016-12-21 23:27:33,481 - Validation loss = 929.410
2016-12-21 23:27:33,481 - Validation accuracy = 0.082
2016-12-21 23:28:02,255 - EPOCH 5 ...
2016-12-21 23:28:02,255 - Validation loss = 322.048
2016-12-21 23:28:02,255 - Validation accuracy = 0.081
2016-12-21 23:28:33,765 - EPOCH 6 ...
2016-12-21 23:28:33,765 - Validation loss = 62.615
2016-12-21 23:28:33,765 - Validation accuracy = 0.064
2016-12-21 23:29:03,515 - EPOCH 7 ...
2016-12-21 23:29:03,515 - Validation loss = 28.383
2016-12-21 23:29:03,515 - Validation accuracy = 0.057
2016-12-21 23:29:31,714 - EPOCH 8 ...
2016-12-21 23:29:31,714 - Validation loss = 19.154
2016-12-21 23:29:31,715 - Validation accuracy = 0.066
2016-12-21 23:29:58,716 - EPOCH 9 ...
2016-12-21 23:29:58,716 - Validation loss = 15.080
2016-12-21 23:29:58,716 - Validation accuracy = 0.064
2016-12-21 23:30:27,351 - EPOCH 10 ...
2016-12-21 23:30:27,351 - Validation loss = 12.431
2016-12-21 23:30:27,352 - Validation accuracy = 0.066
2016-12-21 23:30:54,835 - EPOCH 11 ...
2016-12-21 23:30:54,836 - Validation loss = 10.756
2016-12-21 23:30:54,836 - Validation accuracy = 0.066
2016-12-21 23:31:22,084 - EPOCH 12 ...
2016-12-21 23:31:22,085 - Validation loss = 9.575
2016-12-21 23:31:22,085 - Validation accuracy = 0.066
2016-12-21 23:31:52,247 - EPOCH 13 ...
2016-12-21 23:31:52,247 - Validation loss = 8.594
2016-12-21 23:31:52,247 - Validation accuracy = 0.066
2016-12-21 23:32:22,440 - EPOCH 14 ...
2016-12-21 23:32:22,440 - Validation loss = 7.941
2016-12-21 23:32:22,440 - Validation accuracy = 0.060
"""