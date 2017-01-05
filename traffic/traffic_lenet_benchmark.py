import unittest
import logging.config
from tensorflow.python.framework import dtypes
from .traffic_lenet import Lenet
from .traffic_data import TrafficDataSets
from .traffic_data import DataSet
from .traffic_data import DataSetWithGenerator
from .traffic_data import TrafficDataRealFileProviderAutoSplitValidationData
from .traffic_data import DataSetType
from .traffic_test_data_provider import *
from .traffic_data_enhance import *
import os
logging.config.fileConfig('logging.conf')


def get_and_make_sure_folder_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


class TestLenetBenchmark(unittest.TestCase):
    def test_lenet_original_data(self):
        """
        2017-01-05 11:32:56,256 - EPOCH 99 Validation loss = 4632.268 accuracy = 0.891
        2017-01-05 11:33:29,352 - EPOCH 100 Validation loss = 4487.364 accuracy = 0.892
        2017-01-05 11:33:36,193 - Test loss = 17121.372 accuracy = 0.774
        """
        lenet = Lenet(TrafficDataSets(real_data_provider),
                      name="lenet_original_data",
                      epochs=100, batch_size=500)
        lenet.train()

    def test_lenet_normal_zero_mean_no_grayscale(self):
        """
        2016-12-28 11:32:58,681 - EPOCH 100 ...
        2016-12-28 11:32:58,682 - Validation loss = 31.360
        2016-12-28 11:32:58,682 - Validation accuracy = 0.880
        2016-12-28 11:33:03,870 - Test loss = 119.563
        2016-12-28 11:33:03,870 - Test accuracy = 0.751
        """
        real_data_provider = TrafficDataRealFileProviderAutoSplitValidationData(split_validation_from_train=True, validation_size=0.20)
        _X_train, _y_train = normalise_image_zero_mean(
            real_data_provider_no_shuffer.X_train,
            real_data_provider_no_shuffer.y_train)

        real_data_provider_enhanced_value = TrafficDataProviderAutoSplitValidationData(
            _X_train,
            _y_train,
            real_data_provider_no_shuffer.X_test,
            real_data_provider_no_shuffer.y_test,
            split_validation_from_train=True
        )
        return real_data_provider_enhanced_value
        lenet = Lenet(TrafficDataSets(real_data_provider),
                      name="normal_no_grayscale_Epoch_100_Batch_Size_500_ZeroMean")
        lenet.train()

    def test_lenet_normal_no_grayscale(self):
        """
        2016-12-28 11:32:58,681 - EPOCH 100 ...
        2016-12-28 11:32:58,682 - Validation loss = 31.360
        2016-12-28 11:32:58,682 - Validation accuracy = 0.880
        2016-12-28 11:33:03,870 - Test loss = 119.563
        2016-12-28 11:33:03,870 - Test accuracy = 0.751
        """

        lenet = Lenet(TrafficDataSets(real_data_provider),
                      name="normal_no_grayscale_Epoch_100_Batch_Size_500_ZeroMean")
        lenet.train()

    def test_lenet_normal_no_grayscale_enhanced_with_random_rotate_184700_samples(self):
        """
        2016-12-31 14:15:58,796 - EPOCH 99 Validation loss = 7.603 accuracy = 0.970
        2016-12-31 14:17:03,709 - EPOCH 100 Validation loss = 7.354 accuracy = 0.970
        2016-12-31 14:17:09,565 - Test loss = 84.900 accuracy = 0.816

        2017-01-04 13:24:32,252 - EPOCH 49 Validation loss = 2.133 accuracy = 0.972
        2017-01-04 13:27:30,197 - EPOCH 50 Validation loss = 1.647 accuracy = 0.975
        2017-01-04 13:27:37,678 - Test loss = 29.316 accuracy = 0.832
        """
        lenet = Lenet(TrafficDataSets(real_data_provider_enhanced_with_random_rotate(2), dtype=dtypes.float32, grayscale=False,
                                      training_dataset_factory=normal_dataset_factory,
                                      test_dataset_factory=normal_dataset_factory),
                      name="normal_no_grayscale_Epoch_100_Batch_Size_500_ZeroMean_enhanced_data",
                      epochs=50,
                      batch_size=500)
        lenet.train()

    def test_lenet_normal_no_grayscale_enhanced_data_with_random_zoomin(self):
        """
        017-01-01 11:48:26,154 - training data 334113
        2017-01-01 11:53:27,159 - EPOCH 1 Validation loss = 182.157 accuracy = 0.131
        2017-01-01 11:58:17,694 - EPOCH 2 Validation loss = 5.446 accuracy = 0.035
        2017-01-01 12:03:15,112 - EPOCH 3 Validation loss = 4.031 accuracy = 0.032
        2017-01-01 12:08:11,213 - EPOCH 4 Validation loss = 3.860 accuracy = 0.031
        2017-01-01 12:13:07,066 - EPOCH 5 Validation loss = 3.808 accuracy = 0.031
        2017-01-01 12:18:04,764 - EPOCH 6 Validation loss = 3.785 accuracy = 0.031
        2017-01-01 12:23:04,780 - EPOCH 7 Validation loss = 3.771 accuracy = 0.033
        2017-01-01 12:27:56,787 - EPOCH 8 Validation loss = 3.763 accuracy = 0.033
        2017-01-01 12:32:42,088 - EPOCH 9 Validation loss = 3.758 accuracy = 0.031
        2017-01-01 12:37:56,701 - EPOCH 10 Validation loss = 3.755 accuracy = 0.033

        .....
        2017-01-01 23:32:29,869 - training data 184700
        .....
        2017-01-02 09:53:36,455 - EPOCH 98 Validation loss = 1.000 accuracy = 0.991
        2017-01-02 09:56:22,135 - EPOCH 99 Validation loss = 0.978 accuracy = 0.993
        2017-01-02 09:59:07,160 - EPOCH 100 Validation loss = 0.957 accuracy = 0.992
        2017-01-02 09:59:13,617 - Test loss = 36.678 accuracy = 0.883
        """
        lenet = Lenet(TrafficDataSets(real_data_provider_enhanced_with_random_zoomin(2), dtype=dtypes.float32, grayscale=False,
                                      training_dataset_factory=normal_dataset_factory,
                                      test_dataset_factory=normal_dataset_factory),
                      name="normal_no_grayscale_Epoch_100_Batch_Size_500_ZeroMean_enhanced_data",
                      epochs=50,
                      batch_size=500
        )
        lenet.train()

    def test_lenet_normal_no_grayscale_enhanced_with_random_rotate_zoomin_184700_samples(self):
        """
        2017-01-04 15:59:52,448 - EPOCH 48 Validation loss = 2.011 accuracy = 0.822
        2017-01-04 16:02:25,239 - EPOCH 49 Validation loss = 1.974 accuracy = 0.828
        2017-01-04 16:04:58,320 - EPOCH 50 Validation loss = 1.847 accuracy = 0.834
        2017-01-04 16:05:04,495 - Test loss = 13.297 accuracy = 0.682

        2017-01-05 08:08:51,000 - EPOCH 99 Validation loss = 0.649 accuracy = 0.953
        2017-01-05 08:11:38,264 - EPOCH 100 Validation loss = 0.658 accuracy = 0.953
        2017-01-05 08:11:44,905 - Test loss = 8.400 accuracy = 0.757
        """
        lenet = Lenet(
            TrafficDataSets(real_data_provider_enhanced_with_random_rotate_and_zoomin(2), dtype=dtypes.float32, grayscale=False,
                            training_dataset_factory=normal_dataset_factory,
                            test_dataset_factory=normal_dataset_factory),
            name="normal_no_grayscale_Epoch_100_Batch_Size_500_ZeroMean_enhanced_random_rotate_zoomin",
            epochs=100,
            batch_size=500)
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
        lenet = Lenet(TrafficDataSets(real_data_provider_enhanced_with_random_rotate(), dtype=dtypes.float32, grayscale=True,
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
