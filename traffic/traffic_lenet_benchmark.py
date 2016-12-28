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
    def test_lenet_normal_no_grayscale(self):
        """
        2016-12-28 11:32:58,681 - EPOCH 100 ...
        2016-12-28 11:32:58,682 - Validation loss = 31.360
        2016-12-28 11:32:58,682 - Validation accuracy = 0.880
        2016-12-28 11:33:03,870 - Test loss = 119.563
        2016-12-28 11:33:03,870 - Test accuracy = 0.751
        """
        lenet = Lenet(TrafficDataSets(real_data_provider, dtype=dtypes.float32, grayscale=False,
                                      dataset_factory=normal_dataset_factory))
        lenet.train()


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