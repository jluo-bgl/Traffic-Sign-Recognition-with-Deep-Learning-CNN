import unittest
import logging.config
from tensorflow.python.framework import dtypes
from .traffic_lenet import Lenet
from .traffic_lenet_v2 import LenetV2
from .traffic_lenet_v3 import LenetV3
from .traffic_lenet_v4 import LenetV4
from .traffic_lenet_v5 import LenetV5
from .traffic_lenet_v6 import LenetV6Deep24x96
from .traffic_lenet_v7 import LenetV7LessMaxPooling
from .traffic_lenet_v8_108x200 import LenetV8Deep108x200
from .traffic_data import TrafficDataSets
from .traffic_net_inception import NetInception
from .traffic_data import DataSet
from .traffic_data import DataSetWithGenerator
from .traffic_data import DataSetType
from .traffic_test_data_provider import TrafficDataRealFileProviderAutoSplitValidationData
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
        2017-01-05 23:21:55,745 - EPOCH 9 Validation loss = 0.342 accuracy = 0.930
        2017-01-05 23:22:29,297 - EPOCH 10 Validation loss = 0.363 accuracy = 0.930
        2017-01-05 23:22:35,478 - Test loss = 1.426 accuracy = 0.829
        """
        real_data_provider = TrafficDataRealFileProviderAutoSplitValidationData(
            split_validation_from_train=True, validation_size=0.20)
        lenet = Lenet(TrafficDataSets(real_data_provider),
                      name="lenet_original_data",
                      epochs=1, batch_size=128,
                      variable_mean=0, variable_stddev=0.1,
                      drop_out_keep_prob=1
                      )
        lenet.train()

    def test_lenet_original_data_grayscale(self):
        """
        2017-01-06 22:28:20,266 - EPOCH 9 Validation loss = 0.218 accuracy = 0.960
        2017-01-06 22:28:42,081 - EPOCH 10 Validation loss = 0.222 accuracy = 0.959
        2017-01-06 22:28:45,901 - Test loss = 1.057 accuracy = 0.876
        """
        provider = TrafficDataRealFileProviderAutoSplitValidationData(
            split_validation_from_train=True, validation_size=0.20)
        provider = grayscale(provider)
        lenet = Lenet(TrafficDataSets(provider),
                      name="lenet_original_data",
                      epochs=10, batch_size=128,
                      variable_mean=0, variable_stddev=0.1
                      )
        lenet.train()

    def test_lenet_brightness_contrast_data(self):
        """
        2017-01-05 23:21:55,745 - EPOCH 9 Validation loss = 0.342 accuracy = 0.930
        2017-01-05 23:22:29,297 - EPOCH 10 Validation loss = 0.363 accuracy = 0.930
        2017-01-05 23:22:35,478 - Test loss = 1.426 accuracy = 0.829
        """
        provider = TrafficDataRealFileProviderAutoSplitValidationData(
            split_validation_from_train=True, validation_size=0.20)
        images, labels = enhance_with_tensorflow_brightness_contrast_bulk(provider.X_train, provider.y_train, 2)
        provider = provider.to_other_provider(X_train_overwrite=images, y_train_overwrite=labels)
        lenet = Lenet(TrafficDataSets(provider),
                      name="lenet_original_data",
                      epochs=3, batch_size=128,
                      variable_mean=0, variable_stddev=0.1,
                      drop_out_keep_prob=1
                      )
        lenet.train()

    def test_lenet_original_data_grayscale_v2(self):
        """
        2017-01-06 22:28:20,266 - EPOCH 9 Validation loss = 0.218 accuracy = 0.960
        2017-01-06 22:28:42,081 - EPOCH 10 Validation loss = 0.222 accuracy = 0.959
        2017-01-06 22:28:45,901 - Test loss = 1.057 accuracy = 0.876
        """
        provider = TrafficDataRealFileProviderAutoSplitValidationData(
            split_validation_from_train=True, validation_size=0.20)
        provider = grayscale(provider)
        lenet = LenetV2(TrafficDataSets(provider),
                      name="lenet_original_data",
                      epochs=10, batch_size=128,
                      variable_mean=0, variable_stddev=0.1
                      )
        lenet.train()

    def test_lenet_original_data_grayscale_v3(self):
        """
        2017-01-06 22:28:20,266 - EPOCH 9 Validation loss = 0.218 accuracy = 0.960
        2017-01-06 22:28:42,081 - EPOCH 10 Validation loss = 0.222 accuracy = 0.959
        2017-01-06 22:28:45,901 - Test loss = 1.057 accuracy = 0.876
        """
        provider = TrafficDataRealFileProviderAutoSplitValidationData(
            split_validation_from_train=True, validation_size=0.20)
        provider = grayscale(provider)
        lenet = LenetV3(TrafficDataSets(provider),
                        name="lenet_original_data",
                        epochs=20, batch_size=128,
                        variable_mean=0, variable_stddev=0.1
                        )
        lenet.train()

    def test_lenet_original_data_grayscale_v4(self):
        """
        2017-01-06 22:28:20,266 - EPOCH 9 Validation loss = 0.218 accuracy = 0.960
        2017-01-06 22:28:42,081 - EPOCH 10 Validation loss = 0.222 accuracy = 0.959
        2017-01-06 22:28:45,901 - Test loss = 1.057 accuracy = 0.876
        """
        provider = TrafficDataRealFileProviderAutoSplitValidationData(
            split_validation_from_train=True, validation_size=0.20)
        provider = grayscale(provider)
        lenet = LenetV4(TrafficDataSets(provider),
                        name="lenet_original_data",
                        epochs=10, batch_size=128,
                        variable_mean=0, variable_stddev=0.1
                        )
        lenet.train()

    def test_lenet_original_data_grayscale_v5(self):
        """
        2017-01-06 22:28:20,266 - EPOCH 9 Validation loss = 0.218 accuracy = 0.960
        2017-01-06 22:28:42,081 - EPOCH 10 Validation loss = 0.222 accuracy = 0.959
        2017-01-06 22:28:45,901 - Test loss = 1.057 accuracy = 0.876
        """
        provider = TrafficDataRealFileProviderAutoSplitValidationData(
            split_validation_from_train=True, validation_size=0.20)
        provider = grayscale(provider)
        provider = normalise_image_unit_variance(provider)
        # provider = normalise_image_zero_mean(provider)
        lenet = LenetV5(TrafficDataSets(provider),
                        name="lenet_original_data",
                        epochs=10, batch_size=128,
                        variable_mean=0, variable_stddev=0.1
                        )
        lenet.train()


    def test_lenet_original_data_grayscale_v5_best(self):
        """
        2017-01-06 14:33:14,338 - EPOCH 39 Validation loss = 0.072 accuracy = 0.987
        2017-01-06 14:33:17,597 - EPOCH 40 Validation loss = 0.071 accuracy = 0.988
        2017-01-06 14:33:19,000 - Test loss = 0.483 accuracy = 0.935
        :return:
        """
        provider = TrafficDataRealFileProviderAutoSplitValidationData(
            split_validation_from_train=True, validation_size=0.20)

        images, labels = enhance_with_random_rotate(provider.X_train, provider.y_train, 2)
        provider = provider.to_other_provider(X_train_overwrite=images, y_train_overwrite=labels)
        #     provider = grayscale(provider)
        #     provider = normalise_image_zero_mean(provider)
        provider = normalise_image_unit_variance(provider)
        lenet = LenetV5(TrafficDataSets(provider),
                        name="lenet_original_data",
                        epochs=10, batch_size=128,
                        variable_mean=0, variable_stddev=0.1, learning_rate=0.001,
                        drop_out=0.5
                        )
        lenet.train()

    def test_lenet_original_data_grayscale_inception(self):
        """
        2017-01-06 14:33:14,338 - EPOCH 39 Validation loss = 0.072 accuracy = 0.987
        2017-01-06 14:33:17,597 - EPOCH 40 Validation loss = 0.071 accuracy = 0.988
        2017-01-06 14:33:19,000 - Test loss = 0.483 accuracy = 0.935
        :return:
        """
        provider = TrafficDataRealFileProviderAutoSplitValidationData(
            split_validation_from_train=True, validation_size=0.20)

        # images, labels = enhance_with_random_rotate(provider.X_train, provider.y_train, 2)
        # provider = provider.to_other_provider(X_train_overwrite=images, y_train_overwrite=labels)
        #     provider = grayscale(provider)
        #     provider = normalise_image_zero_mean(provider)
        provider = normalise_image_unit_variance(provider)
        lenet = NetInception(TrafficDataSets(provider),
                             name="lenet_original_data",
                             epochs=10, batch_size=128,
                             variable_mean=0, variable_stddev=0.1, learning_rate=0.001,
                             drop_out_keep_prob=0.5
                            )
        lenet.train()


    def test_lenet_original_data_batch_500(self):
        """
        2017-01-05 18:37:10,487 - EPOCH 99 Validation loss = 107.213 accuracy = 0.069
        2017-01-05 18:37:38,945 - EPOCH 100 Validation loss = 94.099 accuracy = 0.065
        2017-01-05 18:37:44,418 - Test loss = 547.546 accuracy = 0.064
        """
        real_data_provider = TrafficDataRealFileProviderAutoSplitValidationData(
            split_validation_from_train=True, validation_size=0.20)
        lenet = Lenet(TrafficDataSets(real_data_provider),
                      name="lenet_original_data_batch_500",
                      epochs=10, batch_size=500)
        lenet.train()

    def test_lenet_v2_original_data(self):
        """

        """
        real_data_provider = TrafficDataRealFileProviderAutoSplitValidationData(
            split_validation_from_train=True, validation_size=0.20)
        lenet = Lenet(TrafficDataSets(real_data_provider),
                      name="lenet_original_data",
                      epochs=10, batch_size=128)
        lenet.train()

    def test_lenet_normal_zero_mean_no_grayscale(self):
        """
        2017-01-05 23:29:04,073 - EPOCH 9 Validation loss = 0.385 accuracy = 0.886
        2017-01-05 23:29:41,245 - EPOCH 10 Validation loss = 0.358 accuracy = 0.897
        2017-01-05 23:29:48,180 - Test loss = 1.290 accuracy = 0.739
        """
        real_data_provider = TrafficDataRealFileProviderAutoSplitValidationData(
            split_validation_from_train=True, validation_size=0.20)
        real_data_provider = normalise_image_zero_mean(real_data_provider)
        lenet = Lenet(TrafficDataSets(real_data_provider),
                      name="lenet_original_ZeroMean",
                      epochs=10, batch_size=128)
        lenet.train()

    def test_lenet_normal_no_grayscale_enhanced_with_random_rotate_184700_samples(self):
        """
        2016-12-31 14:15:58,796 - EPOCH 99 Validation loss = 7.603 accuracy = 0.970
        2016-12-31 14:17:03,709 - EPOCH 100 Validation loss = 7.354 accuracy = 0.970
        2016-12-31 14:17:09,565 - Test loss = 84.900 accuracy = 0.816

        2017-01-04 13:24:32,252 - EPOCH 49 Validation loss = 2.133 accuracy = 0.972
        2017-01-04 13:27:30,197 - EPOCH 50 Validation loss = 1.647 accuracy = 0.975
        2017-01-04 13:27:37,678 - Test loss = 29.316 accuracy = 0.832

        epochs:10, batch_size=128, stddev=0.1
        2017-01-06 00:00:29,833 - EPOCH 9 Validation loss = 0.291 accuracy = 0.923
        2017-01-06 00:03:30,843 - EPOCH 10 Validation loss = 0.252 accuracy = 0.935
        2017-01-06 00:03:38,386 - Test loss = 1.427 accuracy = 0.775

        epochs=100, batch_size=500
        2017-01-06 13:02:52,200 - EPOCH 99 Validation loss = 0.399 accuracy = 0.956
        2017-01-06 13:05:24,456 - EPOCH 100 Validation loss = 0.409 accuracy = 0.953
        2017-01-06 13:05:31,330 - Test loss = 3.315 accuracy = 0.800
        """
        real_data_provider = TrafficDataRealFileProviderAutoSplitValidationData(
            split_validation_from_train=True, validation_size=0.20)
        images, labels = enhance_with_random_rotate(real_data_provider.X_train, real_data_provider.y_train, 2)
        provider = real_data_provider.to_other_provider(X_train_overwrite=images, y_train_overwrite=labels)
        provider = normalise_image_zero_mean(provider)
        lenet = Lenet(TrafficDataSets(provider),
                        name="normal_no_grayscale_ZeroMean_enhanced_rotate_2",
                        epochs=100, batch_size=500)
        lenet.train()

    def test_lenet_normal_no_grayscale_enhanced_data_with_random_zoomin(self):
        """
        .....
        2017-01-01 23:32:29,869 - training data 184700
        .....
        2017-01-02 09:53:36,455 - EPOCH 98 Validation loss = 1.000 accuracy = 0.991
        2017-01-02 09:56:22,135 - EPOCH 99 Validation loss = 0.978 accuracy = 0.993
        2017-01-02 09:59:07,160 - EPOCH 100 Validation loss = 0.957 accuracy = 0.992
        2017-01-02 09:59:13,617 - Test loss = 36.678 accuracy = 0.883
        """
        real_data_provider = TrafficDataRealFileProviderAutoSplitValidationData(
            split_validation_from_train=True, validation_size=0.20)
        images, labels = enhance_with_random_zoomin(real_data_provider.X_train, real_data_provider.y_train, 2)
        provider = real_data_provider.to_other_provider(X_train_overwrite=images, y_train_overwrite=labels)
        provider = normalise_image_zero_mean(provider)
        lenet = Lenet(TrafficDataSets(provider),
                      name="normal_no_grayscale_ZeroMean_enhanced_zoomin_2",
                      epochs=10, batch_size=128,
                      variable_mean=0, variable_stddev=1.0
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
        real_data_provider = TrafficDataRealFileProviderAutoSplitValidationData(
            split_validation_from_train=True, validation_size=0.20)
        test_image_folder = None
        def keras_training_image_generator_dataset_factory(X, y):
            return DataSetWithGenerator(X, y, DataSetType.Training,
                                        save_to_dir=test_image_folder, save_prefix="training_")

        lenet = Lenet(TrafficDataSets(real_data_provider,
                                      training_dataset_factory=keras_training_image_generator_dataset_factory),
                      name="keras_generator_no_grayscale_Epoch_100_Batch_Size_500_ZeroMean",
                      epochs = 40, batch_size = 128,
                      variable_mean = 0, variable_stddev = 0.1,
                      drop_out_keep_prob = 1,
                      learning_rate=0.001
        )
        lenet.train()
