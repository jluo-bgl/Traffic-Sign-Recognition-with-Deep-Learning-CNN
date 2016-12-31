import unittest
import logging.config
from tensorflow.python.framework import dtypes
from .traffic_lenet import Lenet
from .traffic_data import TrafficDataSets
from .traffic_data import DataSet
from .traffic_data import DataSetWithGenerator
from .traffic_data import TrafficDataRealFileProvider
from .traffic_data import DataSetType
from .traffic_test_data_provider import *
import os
logging.config.fileConfig('logging.conf')

normal_dataset_factory = lambda X, y, dtype, grayscale: DataSet(X, y, 500, dtype, grayscale)


def get_and_make_sure_folder_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


class TestLenetBenchmark(unittest.TestCase):
    def test_lenet_normal_no_grayscale(self):
        """
        2016-12-28 11:32:58,681 - EPOCH 100 ...
        2016-12-28 11:32:58,682 - Validation loss = 31.360
        2016-12-28 11:32:58,682 - Validation accuracy = 0.880
        2016-12-28 11:33:03,870 - Test loss = 119.563
        2016-12-28 11:33:03,870 - Test accuracy = 0.751
        """
        lenet = Lenet(TrafficDataSets(real_data_provider, dtype=dtypes.float32, grayscale=False,
                                      training_dataset_factory=normal_dataset_factory,
                                      test_dataset_factory=normal_dataset_factory),
                      name="normal_no_grayscale_Epoch_100_Batch_Size_500_ZeroMean")
        lenet.train()

    def test_lenet_normal_no_grayscale_enhanced_data(self):
        """
        2016-12-31 14:15:58,796 - EPOCH 99 Validation loss = 7.603 accuracy = 0.970
        2016-12-31 14:17:03,709 - EPOCH 100 Validation loss = 7.354 accuracy = 0.970
        2016-12-31 14:17:09,565 - Test loss = 84.900 accuracy = 0.816
        """
        lenet = Lenet(TrafficDataSets(real_data_provider_enhanced, dtype=dtypes.float32, grayscale=False,
                                      training_dataset_factory=normal_dataset_factory,
                                      test_dataset_factory=normal_dataset_factory),
                      name="normal_no_grayscale_Epoch_100_Batch_Size_500_ZeroMean_enhanced_data")
        lenet.train()

    def test_lenet_normal_grayscale(self):
        """
        2016-12-28 14:00:46,518 - EPOCH 100 ...
        2016-12-28 14:00:46,518 - Validation loss = 15.567
        2016-12-28 14:00:46,518 - Validation accuracy = 0.888
        2016-12-28 14:00:50,919 - Test loss = 61.315
        2016-12-28 14:00:50,919 - Test accuracy = 0.768
        """
        lenet = Lenet(TrafficDataSets(real_data_provider, dtype=dtypes.float32, grayscale=True,
                                      training_dataset_factory=normal_dataset_factory,
                                      test_dataset_factory=normal_dataset_factory),
                      name="normal_grayscale_Epoch_100_Batch_Size_500_ZeroMean")
        lenet.train()

    def test_lenet_normal_grayscale_enhanced_data(self):
        """
        2016-12-31 16:17:09,525 - EPOCH 99 Validation loss = 4.134 accuracy = 0.959
        2016-12-31 16:17:50,152 - EPOCH 100 Validation loss = 4.120 accuracy = 0.958
        2016-12-31 16:17:53,932 - Test loss = 34.247 accuracy = 0.817
        """
        lenet = Lenet(TrafficDataSets(real_data_provider_enhanced, dtype=dtypes.float32, grayscale=True,
                                      training_dataset_factory=normal_dataset_factory,
                                      test_dataset_factory=normal_dataset_factory),
                      name="normal_grayscale_Epoch_100_Batch_Size_500_ZeroMean")
        lenet.train()


    def test_lenet_keras_generator_no_grayscale(self):
        """
        couldn't make it work, accuracy always between 0.03 to 0.06
        """
        # test_image_folder = get_and_make_sure_folder_exists("./lenet_keras_generator")
        test_image_folder = None
        def keras_training_image_generator_dataset_factory(X, y, dtype, grayscale):
            return DataSetWithGenerator(X, y, 500, DataSetType.Training, dtypes.uint8, grayscale=False,
                                        save_to_dir=test_image_folder, save_prefix="training_")

        def keras_test_image_generator_dataset_factory(X, y, dtype, grayscale):
            return DataSetWithGenerator(X, y, 500, DataSetType.TestAndValudation, dtypes.uint8, grayscale=False,
                                        save_to_dir=test_image_folder, save_prefix="test_validation_")

        lenet = Lenet(TrafficDataSets(real_data_provider, dtype=dtypes.uint8, grayscale=False,
                                      training_dataset_factory=keras_training_image_generator_dataset_factory,
                                      test_dataset_factory=keras_test_image_generator_dataset_factory),
                      name="keras_generator_no_grayscale_Epoch_100_Batch_Size_500_ZeroMean")
        lenet.train()
