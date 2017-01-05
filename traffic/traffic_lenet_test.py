import unittest
import numpy.testing
import tensorflow as tf
import logging.config
from .traffic_lenet import Lenet
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

