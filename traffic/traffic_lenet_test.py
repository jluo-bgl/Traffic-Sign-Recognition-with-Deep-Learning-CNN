import unittest
import logging.config
from tensorflow.python.framework import dtypes
from .traffic_lenet import Lenet
from .traffic_data import TrafficDataSets
from .traffic_data import DataSet
from .traffic_data import DataSetWithGenerator
from .traffic_data import TrafficDataRealFileProvider
logging.config.fileConfig('logging.conf')

real_data_provider = TrafficDataRealFileProvider(split_validation_from_train=True)

normal_dataset_factory = lambda X, y, dtype, grayscale: DataSet(X, y, dtype, grayscale)
keras_image_generator_dataset_factory = lambda X, y, dtype, grayscale: DataSetWithGenerator(X, y, dtype, grayscale)


class TestLenet(unittest.TestCase):
    def test_init(self):
        lenet = Lenet(TrafficDataSets(real_data_provider, grayscale=False))
        self.assertEqual(lenet.x._shape[3], 3, "placeholder for input x should have same color channel as input data")
        lenet = Lenet(TrafficDataSets(real_data_provider, dtype=dtypes.float32, grayscale=True))
        self.assertEqual(lenet.x._shape[3], 1, "placeholder for input x should have same color channel as input data")
