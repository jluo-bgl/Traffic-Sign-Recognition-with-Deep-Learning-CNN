import tensorflow as tf
from tensorflow.contrib.layers import flatten
from .traffic_data import TrafficDataSets
from .data_explorer import TrainingPlotter
from .data_explorer import SignNames
import numpy as np
from enum import Enum
import logging.config
logging.config.fileConfig('logging.conf')


class DropOutPosition(Enum):
    AfterFirstConv = 1
    AfterSecondConv = 2
    AfterAllFullConnectedLayers = 3

class Lenet(object):

    def __init__(self, traffic_dataset, name, show_plot_window=False, sign_names = SignNames("signnames.csv"),
                 epochs=100, batch_size=500,
                 variable_mean=0., variable_stddev=1., learning_rate=0.001, drop_out_keep_prob=0.5):
        self.sign_names = sign_names
        self.file_name = './model_comparison/Lenet_{}_{}.png'.format(name, TrainingPlotter.now_as_str())
        self.file_name_model = './model_comparison/Lenet_{}_{}.model'.format(name, TrainingPlotter.now_as_str())
        self.file_name_confusion_matrix = './model_comparison/Lenet_confusion_matrix_{}_{}.png'\
            .format(name, TrainingPlotter.now_as_str())
        self.file_name_wrong_predicts = './model_comparison/Lenet_wrong_predicts_{}_{}.png'\
            .format(name, TrainingPlotter.now_as_str())
        title = "{}_{}_epochs_{}_batch_size_{}_learning_rate_{}_keep_prob_{}_variable_stddev_{}"\
            .format(self.__class__.__name__, name, epochs, batch_size,
                    learning_rate, drop_out_keep_prob, variable_stddev)
        self.plotter = TrainingPlotter(title,
                                       self.file_name,
                                       show_plot_window=show_plot_window)
        self.epochs = epochs
        self.batch_size = batch_size
        self.label_size = TrafficDataSets.NUMBER_OF_CLASSES

        self.traffic_datas = traffic_dataset

        self.variable_mean = variable_mean
        self.variable_stddev = variable_stddev

        logging.info("training data {}".format(len(traffic_dataset.train.images)))

        self.session = None

        # consists of 32x32xcolor_channel
        color_channel = traffic_dataset.train.images.shape[3]
        self.x = tf.placeholder(tf.float32, (None, 32, 32, color_channel))

        self.y = tf.placeholder(tf.float32, (None, self.label_size))
        self.keep_prob = tf.placeholder(tf.float32)
        self.drop_out_keep_prob = drop_out_keep_prob
        self.network = Lenet._LeNet(self, self.x, color_channel, variable_mean, variable_stddev)

        self.prediction_softmax = tf.nn.softmax(self.network)

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.network, self.y))
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.opt.minimize(self.loss_op)
        self.correct_prediction = tf.equal(tf.argmax(self.network, 1), tf.argmax(self.y, 1))
        self.accuracy_op = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    # LeNet architecture:
    # INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
    # create the LeNet and return the result of the last fully connected layer.
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

        conv2 = tf.nn.dropout(conv2, self.keep_prob)

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
        steps_per_epoch = dataset.num_examples // self.batch_size
        num_examples = steps_per_epoch * self.batch_size
        total_acc, total_loss = 0, 0
        sess = self.session
        # tf.get_default_session()
        for step in range(steps_per_epoch):
            batch_x, batch_y = dataset.next_batch(self.batch_size)
            loss, acc = sess.run([self.loss_op, self.accuracy_op], feed_dict={self.x: batch_x, self.y: batch_y,
                                                                         self.keep_prob: 1.0})
            total_acc += (acc * batch_x.shape[0])
            total_loss += (loss * batch_x.shape[0])
        return total_loss / num_examples, total_acc / num_examples

    def test_data(self, dataset):
        steps_per_epoch = dataset.num_examples // self.batch_size
        num_examples = steps_per_epoch * self.batch_size
        total_acc, total_loss = 0, 0
        total_predict, total_actual = [], []
        wrong_predict_images = []
        sess = self.session
        # tf.get_default_session()
        for step in range(steps_per_epoch):
            batch_x, batch_y = dataset.next_batch(self.batch_size)
            loss, acc, predict, actual = sess.run(
                [self.loss_op, self.accuracy_op, tf.argmax(self.network, 1), tf.argmax(self.y, 1)],
                feed_dict={self.x: batch_x, self.y: batch_y,
                self.keep_prob: 1.0})
            total_acc += (acc * batch_x.shape[0])
            total_loss += (loss * batch_x.shape[0])
            total_predict = np.append(total_predict, predict)
            total_actual = np.append(total_actual, actual)
            for index in range(len(predict)):
                if predict[index] != actual[index]:
                    wrong_predict_images.append(batch_x[index])

        return total_loss / num_examples, total_acc / num_examples, total_predict, total_actual, wrong_predict_images

    def train(self):
        saver = tf.train.Saver()
        if self.session is not None:
            self.session.close()

        self.session = tf.Session()
        # with tf.Session() as sess:
        sess = self.session
        if sess is not None:
            self.session = sess
            sess.run(tf.initialize_all_variables())
            steps_per_epoch = self.traffic_datas.train.num_examples // self.batch_size
            num_examples = steps_per_epoch * self.batch_size
            # Train model
            for i in range(self.epochs):
                self.traffic_datas.train.shuffle()
                total_tran_loss = 0.0
                total_tran_acc = 0.0
                for step in range(steps_per_epoch):
                    batch_x, batch_y = self.traffic_datas.train.next_batch(self.batch_size)
                    _, tran_loss, tran_acc = sess.run(
                        [self.train_op, self.loss_op, self.accuracy_op],
                        feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: self.drop_out_keep_prob})
                    total_tran_loss += (tran_loss * batch_x.shape[0])
                    total_tran_acc += (tran_acc * batch_x.shape[0])

                total_tran_loss = total_tran_loss / num_examples
                total_tran_acc = total_tran_acc / num_examples
                val_loss, val_acc = self.eval_data(self.traffic_datas.validation)
                logging.info("EPOCH {} training loss = {:.3f} accuracy = {:.3f} Validation loss = {:.3f} accuracy = {:.3f}"
                             .format(i + 1, total_tran_loss, total_tran_acc, val_loss, val_acc))
                self.plotter.add_loss_accuracy_to_plot(i, total_tran_loss, total_tran_acc, val_loss, val_acc, redraw=True)

            saver.save(sess, self.file_name_model)
            logging.info("Model saved into {}".format(self.file_name_model))

            # Evaluate on the test data
            test_loss, test_acc, total_predict, total_actual, wrong_predict_images = self.test_data(self.traffic_datas.test)
            logging.info("Test loss = {:.3f} accuracy = {:.3f}".format(test_loss, test_acc))
            self.plotter.plot_confusion_matrix(
                total_actual, total_predict, self.sign_names.names()).savefig(self.file_name_confusion_matrix)
            self.plotter.combine_images(wrong_predict_images, self.file_name_wrong_predicts)

        self.plotter.safe_shut_down()

    def predict_images(self, images):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('.'))
            return sess.run(self.prediction_softmax, feed_dict={self.x: images, self.keep_prob: 1.0})

