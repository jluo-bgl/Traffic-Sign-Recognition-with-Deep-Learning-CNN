import unittest
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from .traffic_data import TrafficDataSets
from .data_explorer import TrainingPlotter
import logging.config
logging.config.fileConfig('logging.conf')

class Lenet(object):

    def __init__(self):
        self.plotter = TrainingPlotter("Lenet Epoch_10 Batch_Size_50",
                                       './model_comparison/Lenet_Epoch_10_Batch_Size_50_{}.png'.format(TrainingPlotter.now_as_str()),
                                       show_plot_window=False)
        self.EPOCHS = 100
        self.BATCH_SIZE = 500
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
        fc1_shape = (fc1.get_shape().as_list()[-1], 512)

        fc1_W = tf.Variable(tf.truncated_normal(shape=(fc1_shape)))
        fc1_b = tf.Variable(tf.zeros(512))
        fc1 = tf.matmul(fc1, fc1_W) + fc1_b
        fc1 = tf.nn.relu(fc1)

        fc2_W = tf.Variable(tf.truncated_normal(shape=(512, self.LABEL_SIZE)))
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
                    self.plotter.add_loss_accuracy_to_plot(i, loss, None, None, None, redraw=False)

                val_loss, val_acc = self.eval_data(self.traffic_datas.validation)
                logging.info("EPOCH {} ...".format(i + 1))
                logging.info("Validation loss = {:.3f}".format(val_loss))
                logging.info("Validation accuracy = {:.3f}".format(val_acc))
                self.plotter.add_loss_accuracy_to_plot(i, loss, None, val_loss, val_acc, redraw=True)

            # Evaluate on the test data
            test_loss, test_acc = self.eval_data(self.traffic_datas.test)
            logging.info("Test loss = {:.3f}".format(test_loss))
            logging.info("Test accuracy = {:.3f}".format(test_acc))

        self.plotter.safe_shut_down()


class TestLenet(unittest.TestCase):
    def test_lenet(self):
        Lenet().train()


"""
2016-12-22 10:09:20,270 - EPOCH 1 ...
2016-12-22 10:09:20,270 - Validation loss = 8.590
2016-12-22 10:09:20,270 - Validation accuracy = 0.032
2016-12-22 10:10:01,254 - EPOCH 2 ...
2016-12-22 10:10:01,254 - Validation loss = 4.388
2016-12-22 10:10:01,254 - Validation accuracy = 0.054
2016-12-22 10:10:38,479 - EPOCH 3 ...
2016-12-22 10:10:38,479 - Validation loss = 3.895
2016-12-22 10:10:38,480 - Validation accuracy = 0.055
2016-12-22 10:11:16,763 - EPOCH 4 ...
2016-12-22 10:11:16,763 - Validation loss = 3.749
2016-12-22 10:11:16,763 - Validation accuracy = 0.058
2016-12-22 10:11:54,226 - EPOCH 5 ...
2016-12-22 10:11:54,226 - Validation loss = 3.683
2016-12-22 10:11:54,226 - Validation accuracy = 0.058
2016-12-22 10:12:31,134 - EPOCH 6 ...
2016-12-22 10:12:31,134 - Validation loss = 3.647
2016-12-22 10:12:31,134 - Validation accuracy = 0.058
2016-12-22 10:13:07,885 - EPOCH 7 ...
2016-12-22 10:13:07,885 - Validation loss = 3.631
2016-12-22 10:13:07,886 - Validation accuracy = 0.059
2016-12-22 10:13:39,800 - EPOCH 8 ...
2016-12-22 10:13:39,800 - Validation loss = 3.618
2016-12-22 10:13:39,800 - Validation accuracy = 0.059
2016-12-22 10:14:16,385 - EPOCH 9 ...
2016-12-22 10:14:16,385 - Validation loss = 3.611
2016-12-22 10:14:16,385 - Validation accuracy = 0.059
2016-12-22 10:14:52,117 - EPOCH 10 ...
2016-12-22 10:14:52,117 - Validation loss = 3.621
2016-12-22 10:14:52,117 - Validation accuracy = 0.059
2016-12-22 10:14:59,254 - Test loss = 3.495
2016-12-22 10:14:59,255 - Test accuracy = 0.061
"""