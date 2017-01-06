import tensorflow as tf
from tensorflow.contrib.layers import flatten
from .traffic_data import TrafficDataSets
from .data_explorer import TrainingPlotter
import logging.config
logging.config.fileConfig('logging.conf')

class LenetV2(object):

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
        self.network = LenetV2._LeNet(self, self.x, color_channel)

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.network, self.y))
        self.opt = tf.train.AdamOptimizer()
        self.train_op = self.opt.minimize(self.loss_op)

    # LeNet architecture:
    # INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
    # create the LeNet and return the result of the last fully connected layer.
    def _LeNet(self, x, color_channel):
        # Hyperparameters
        mu = 0
        sigma = 0.1

        # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, color_channel, 6), mean=mu, stddev=sigma))
        conv1_b = tf.Variable(tf.zeros(6))
        conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

        # SOLUTION: Activation.
        conv1 = tf.nn.relu(conv1)

        # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
        conv2_b = tf.Variable(tf.zeros(16))
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

        # SOLUTION: Activation.
        conv2 = tf.nn.relu(conv2)

        # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
        fc0 = flatten(conv2)

        # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
        fc1_b = tf.Variable(tf.zeros(120))
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b

        # SOLUTION: Activation.
        fc1 = tf.nn.relu(fc1)

        # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
        fc2_b = tf.Variable(tf.zeros(84))
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b

        # SOLUTION: Activation.
        fc2 = tf.nn.relu(fc2)

        # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
        fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
        fc3_b = tf.Variable(tf.zeros(43))
        logits = tf.matmul(fc2, fc3_W) + fc3_b

        return logits

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
        correct_prediction = tf.equal(tf.argmax(self.network, 1), tf.argmax(self.y, 1))
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
