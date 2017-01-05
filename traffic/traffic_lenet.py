import tensorflow as tf
from tensorflow.contrib.layers import flatten
from .traffic_data import TrafficDataSets
from .data_explorer import TrainingPlotter
import logging.config
logging.config.fileConfig('logging.conf')

class Lenet(object):

    def __init__(self, traffic_dataset, name, epochs=100, batch_size=500):
        self.plotter = TrainingPlotter("Lenet " + name,
                                       './model_comparison/Lenet_{}_{}.png'.format(name, TrainingPlotter.now_as_str()),
                                       show_plot_window=False)
        self.epochs = epochs
        self.batch_size = batch_size
        self.label_size = TrafficDataSets.NUMBER_OF_CLASSES

        self.traffic_datas = traffic_dataset

        logging.info("training data {}".format(len(traffic_dataset.train.images)))

        # consists of 32x32xcolor_channel
        color_channel = traffic_dataset.train.images.shape[3]
        self.x = tf.placeholder(tf.float32, (None, 32, 32, color_channel))

        self.y = tf.placeholder(tf.float32, (None, self.label_size))
        self.fc2 = Lenet._LeNet(self, self.x, color_channel)

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.fc2, self.y))
        self.opt = tf.train.AdamOptimizer()
        self.train_op = self.opt.minimize(self.loss_op)

    # LeNet architecture:
    # INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
    # create the LeNet and return the result of the last fully connected layer.
    def _LeNet(self, x, color_channel):
        # x is 32, 32, 3
        # Reshape from 2D to 4D. This prepares the data for
        # convolutional and pooling layers.
        # x = tf.reshape(x, (-1, 32, 32, 3))
        # Pad 0s to 32x32. Centers the digit further.
        # Add 2 rows/columns on each side for height and width dimensions.
        # x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")

        # 28x28x6
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, color_channel, 6)))
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

        fc2_W = tf.Variable(tf.truncated_normal(shape=(512, self.label_size)))
        fc2_b = tf.Variable(tf.zeros(self.label_size))
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
        correct_prediction = tf.equal(tf.argmax(self.fc2, 1), tf.argmax(self.y, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        steps_per_epoch = dataset.num_examples // self.batch_size
        num_examples = steps_per_epoch * self.batch_size
        total_acc, total_loss = 0, 0
        sess = tf.get_default_session()
        for step in range(steps_per_epoch):
            batch_x, batch_y = dataset.next_batch(self.batch_size)
            loss, acc = sess.run([self.loss_op, accuracy_op], feed_dict={self.x: batch_x, self.y: batch_y})
            total_acc += (acc * batch_x.shape[0])
            total_loss += (loss * batch_x.shape[0])
        return total_loss / num_examples, total_acc / num_examples

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            steps_per_epoch = self.traffic_datas.train.num_examples // self.batch_size

            # Train model
            for i in range(self.epochs):
                for step in range(steps_per_epoch):
                    batch_x, batch_y = self.traffic_datas.train.next_batch(self.batch_size)
                    loss = sess.run(self.train_op, feed_dict={self.x: batch_x, self.y: batch_y})
                    self.plotter.add_loss_accuracy_to_plot(i, loss, None, None, None, redraw=False)

                val_loss, val_acc = self.eval_data(self.traffic_datas.validation)
                logging.info("EPOCH {} Validation loss = {:.3f} accuracy = {:.3f}".format(i + 1, val_loss, val_acc))
                self.plotter.add_loss_accuracy_to_plot(i, loss, None, val_loss, val_acc, redraw=True)

            # Evaluate on the test data
            test_loss, test_acc = self.eval_data(self.traffic_datas.test)
            logging.info("Test loss = {:.3f} accuracy = {:.3f}".format(test_loss, test_acc))

        self.plotter.safe_shut_down()
