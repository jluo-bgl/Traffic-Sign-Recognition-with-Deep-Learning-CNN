import unittest
import numpy.testing
import tensorflow as tf
import logging.config
from .traffic_lenet import Lenet, DropOutPosition
from .traffic_data import TrafficDataSets
from .traffic_test_data_provider import real_data_provider

logging.config.fileConfig('logging.conf')

class TestLenet(unittest.TestCase):
    def test_color_channel_no_grayscale_should_equals_to_3(self):
        lenet = Lenet(TrafficDataSets(real_data_provider), name="test init")
        self.assertEqual(lenet.x._shape[3], 3, "placeholder for input x should have same color channel as input data")

    def test_color_channel_grayscale_should_equals_to_1(self):
        lenet = Lenet(TrafficDataSets(real_data_provider), name="test init")
        self.assertEqual(lenet.x._shape[3], 1, "placeholder for input x should have same color channel as input data")

    def test_argmax(self):
        one_host = [[0., 1.], [0., 1.], [1., 0.]]
        result = tf.argmax(one_host, 1).eval(session=tf.Session())
        numpy.testing.assert_allclose(result, [1, 1, 0], err_msg="return the index of the max value in a tensor")

    def test_truncated_normal(self):
        normal = tf.truncated_normal(shape=(200, 300), mean=1, stddev=1).eval(session=tf.Session())
        numpy.testing.assert_allclose([numpy.sum(normal) / (200 * 300)], [1], rtol=0.1)
        print(numpy.min(normal))
        print(numpy.max(normal))
        print(normal)
        normal = tf.truncated_normal(shape=(2, 3), mean=0, stddev=0.1).eval(session=tf.Session())
        print(normal)
        normal = tf.truncated_normal(shape=(2, 3), mean=0, stddev=0.5).eval(session=tf.Session())
        print(numpy.sum(normal))
        print(normal)
        normal = tf.truncated_normal(shape=(2, 3), mean=0, stddev=0.).eval(session=tf.Session())
        print(normal)

    def test_saver(self):
        old_weights, old_bias, new_weights, new_bias = None, None, None, None
        save_file = 'test_data/model.ckpt'
        weights = tf.Variable(tf.truncated_normal([2, 3]))
        bias = tf.Variable(tf.truncated_normal([3]))
        print('Save Weights: {}'.format(weights.name))
        print('Save Bias: {}'.format(bias.name))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # Initialize all the Variables
            sess.run(tf.initialize_all_variables())

            # Show the values of weights and bias
            old_weights = sess.run(weights)
            old_bias = sess.run(bias)
            print('Weights:', old_weights)
            print('Bias:', old_bias)

            # Save the model
            saver.save(sess, save_file)
            # Remove the previous weights and bias

        tf.reset_default_graph()
        weights = tf.Variable(tf.truncated_normal([2, 3]))
        bias = tf.Variable(tf.truncated_normal([3]))
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Load the weights and bias
            saver.restore(sess, save_file)

            # Show the values of weights and bias
            new_weights = sess.run(weights)
            new_bias = sess.run(bias)
            print('Weights:', new_weights)
            print('Bias:', new_bias)

        numpy.testing.assert_allclose(old_weights, new_weights)
        numpy.testing.assert_allclose(old_bias, new_bias)

    def test_in_operation_dropout(self):
        positions = [DropOutPosition.AfterAllFullConnectedLayers, DropOutPosition.AfterFirstConv]
        assert DropOutPosition.AfterAllFullConnectedLayers in positions
        assert DropOutPosition.AfterSecondConv not in positions

