import unittest
from pandas import DataFrame
import pandas as pd
from .data_explorer import SignNames
from .data_explorer import DataExplorer
from .traffic_data import TrafficDataSets
from .traffic_data import TrafficDataProviderAutoSplitValidationData
from .traffic_data import TrafficDataRealFileProviderAutoSplitValidationData
from .traffic_data import DataSetWithGenerator
from .traffic_data import DataSetType
from .traffic_data_enhance import *
from .traffic_data_enhance import _enhance_one_image_randomly
from .traffic_data_enhance import _zoomin_image_randomly
from .traffic_data_enhance import _enhance_one_image_with_random_funcs
from .traffic_data_enhance import _image_grayscale
from .traffic_data_enhance import _normalise_image_zero_mean
from .traffic_data_enhance import _normalise_image, _normalise_image_whitening, \
    _enhance_one_image_with_tensorflow_random_operations
from .data_explorer import TrainingPlotter
from tensorflow.python.framework import dtypes
import pickle
import numpy.testing
import os
import numpy as np
from .traffic_test_data_provider import real_data_provider
from .traffic_test_data_provider import real_data_provider_no_shuffer
from .traffic_test_data_provider import clear_subset_data_provider


class TestTrafficDataProvider(unittest.TestCase):
    def test_validation_split(self):
        provider = TrafficDataRealFileProviderAutoSplitValidationData()
        self.assertEqual(len(provider.X_train), 31367, "80% of training data")
        self.assertEqual(len(provider.X_validation), 7842, "20% of validation data")
        self.assertEqual(len(provider.X_test), 12630, "separated test data")

        provider = TrafficDataRealFileProviderAutoSplitValidationData(split_validation_from_train=True,
                                                                      validation_size=0.30)
        self.assertEqual(len(provider.X_train), 27446, "70% of training data")
        self.assertEqual(len(provider.X_validation), 11763, "30% of validation data")
        self.assertEqual(len(provider.X_test), 12630, "separated test data")

    def test_save_load_file(self):
        provider = real_data_provider
        provider.save_to_file("./test_data/provider_save_file.p")
        loaded_provider = TrafficDataProvider.load_from_file("./test_data/provider_save_file.p")
        self.assertEqual(len(provider.X_train), len(loaded_provider.X_train))
        self.assertEqual(len(provider.y_train), len(loaded_provider.y_train))
        self.assertEqual(len(provider.X_validation), len(loaded_provider.X_validation))
        self.assertEqual(len(provider.y_validation), len(loaded_provider.y_validation))
        self.assertEqual(len(provider.X_test), len(loaded_provider.X_test))
        self.assertEqual(len(provider.y_test), len(loaded_provider.y_test))

class TestTrafficDataSets(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.training_file = "train.p"
        self.testing_file = "test.p"
        self.traffic_datasets = TrafficDataSets(real_data_provider,
                                                one_hot_encode = False)

    def test_init(self):
        self.assertTrue(self.traffic_datasets.test is not None)
        self.assertTrue(self.traffic_datasets.validation is not None)
        self.assertTrue(self.traffic_datasets.train is not None)
        self.assertEqual(self.traffic_datasets.NUMBER_OF_CLASSES, 43, "we have 43 kind of traffic signs")
        self.assertEqual(self.traffic_datasets.train.num_examples, 31367, "70% of training data")
        self.assertEqual(self.traffic_datasets.validation.num_examples, 7842, "30% of validation data")
        self.assertEqual(self.traffic_datasets.test.num_examples, 12630, "30% of test data")

    def test_train_validation_split_follows_distribution(self):
        sign_names = SignNames("signnames.csv")
        datasets = self.traffic_datasets
        distribution = DataExplorer._data_distribution(datasets.validation.labels, sign_names)
        self.assertEqual(len(distribution), TrafficDataSets.NUMBER_OF_CLASSES, "all samples should exists in validation set")
        print(distribution)


def get_and_make_sure_folder_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


class TestTrafficDataSet(unittest.TestCase):

    def test_data_generator_factory(self):
        test_image_folder = get_and_make_sure_folder_exists("./test_data_generator_factory")
        datagen = DataSetWithGenerator._training_data_generator_factory()
        x = clear_subset_data_provider.X_train
        y = clear_subset_data_provider.y_train
        DataExplorer._sample(x, y, slice(0, len(y)), SignNames("signnames.csv")).savefig(test_image_folder + "/original.png")
        i = 0
        for batch in datagen.flow(x, y, batch_size=6,
                                  save_to_dir=test_image_folder, save_prefix='traffic', save_format='jpeg'):
            i += 1
            if i > 20:
                break  # otherwise the generator would loop indefinitely

    def test_next_batch_should_return_X_y_also_save_files_into_folder(self):
        test_image_folder = get_and_make_sure_folder_exists("./test_next_batch_with_generator")
        x = clear_subset_data_provider.X_train
        y = clear_subset_data_provider.y_train
        DataExplorer._sample(x, y, slice(0, len(y)), SignNames("signnames.csv")).savefig(
            test_image_folder + "/original.png")
        dataset = DataSetWithGenerator(x, y, dataset_type=DataSetType.TestAndValudation,
                                       save_to_dir=test_image_folder,
                                       save_prefix='test_next_batch_generator')
        for i in range(20):
            batch = dataset.next_batch(500)
            self.assertEqual(type(batch), tuple)
            self.assertEqual(len(batch), 2)

    def test_one_hot_encode(self):
        labels = np.array([1, 1, 0])
        one_hot = TrafficDataSets.dense_to_one_hot(labels, 2)
        numpy.testing.assert_allclose(one_hot, [[0., 1.], [0., 1.], [1., 0.]])


class TestTrafficDataEnhancement(unittest.TestCase):
    def test_enhance(self):
        features = real_data_provider_no_shuffer.X_train
        lables = real_data_provider_no_shuffer.y_train
        self.assertTrue(len(features) < 40000, "original data has less then 40K features")

        features, labels = enhance_with_random_rotate(real_data_provider_no_shuffer.X_train,
                                                                             real_data_provider_no_shuffer.y_train,
                                                                             1)
        self.assertTrue(len(features) > 70000, "enhanced data has more than 70K features")

    def test_enhance_one_image_with_random_funcs(self):
        def enhance1(image, how_many_to_generate):
            self.assertEqual(image, [1])
            self.assertEqual(how_many_to_generate, 1)
            return [10]

        def enhance2(image, how_many_to_generate):
            self.assertEqual(image, [1])
            self.assertEqual(how_many_to_generate, 1)
            return [20]

        images = _enhance_one_image_with_random_funcs([enhance1, enhance2])([1], 4)
        self.assertTrue("10" in images.__str__())
        self.assertTrue("20" in images.__str__())
        self.assertEqual(len(images), 4)
        print(images)

    def test_enhance_with_function(self):
        def enhance1(image, how_many_to_generate):
            return map(lambda index: [image[0] * 10 + how_many_to_generate], range(how_many_to_generate))

        images, labels = enhance_with_function(
            np.array([[10], [20], [30]]), np.array([0, 1, 1]), 2, enhance1)
        print(images, labels)
        self.assertCountEqual([0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], labels)
        self.assertCountEqual(
            np.array([[10], [20], [30], [104], [104], [104], [104], [202], [202], [302], [302]]),
            images,
            "10 will count 4 times as it has only 1 class. the total number of result will be (1+1)*2classes*2Ratio+3"
        )



    def test_random_zoomin(self):
        original = clear_subset_data_provider.X_train[0]
        label = clear_subset_data_provider.y_train[0]
        all_images = []
        all_labels = []
        all_images.append(original)
        all_labels.append(label)
        for index in range(0, 10):
            all_images.append(_zoomin_image_randomly(original))
            all_labels.append(label)

        plt = DataExplorer._sample(all_images, all_labels, slice(0, 11), SignNames("signnames.csv"))
        test_image_folder = get_and_make_sure_folder_exists("./test_data_enhancement")
        plt.savefig(test_image_folder + "/random_zoomin.png")

    def test_enhance_one_image_randomly(self):
        original = clear_subset_data_provider.X_train[0]
        label = clear_subset_data_provider.y_train[0]
        all_images = []
        all_labels = []
        all_images.append(original)
        all_labels.append(label)
        images, labels = _enhance_one_image_randomly(original, label, 9)
        for i in images:
            all_images.append(i)
        for l in labels:
            all_labels.append(l)

        plt = DataExplorer._sample(all_images, all_labels, slice(0, len(all_images)), SignNames("signnames.csv"))
        test_image_folder = get_and_make_sure_folder_exists("./test_data_enhancement")
        plt.savefig(test_image_folder + "/random_generator.png")

    def test_enhance_infinite(self):
        test_image_folder = get_and_make_sure_folder_exists("./test_data_enhancement")

        images_to_enhance = real_data_provider_no_shuffer.X_train[0:1000:200]
        labels_to_enhance = real_data_provider_no_shuffer.y_train[0:1000:200]

        plt = DataExplorer._sample(images_to_enhance, labels_to_enhance, slice(0, len(images_to_enhance)), SignNames("signnames.csv"))
        plt.savefig(test_image_folder + "/random_generator_infinite_original.png")

        images, labels = enhance_with_random_zoomin(images_to_enhance, labels_to_enhance, 3)
        plt = DataExplorer._sample(images, labels, slice(0, len(images)), SignNames("signnames.csv"))
        plt.savefig(test_image_folder + "/random_generator_infinite.png")

    def test_gray_scale_should_contains_one_chanel(self):
        images = np.array([[[[1, 1, 1], [255, 255, 255], [128, 128, 128]]]])
        self.assertEqual(images.shape[3], 3)
        grayscale_images = _image_grayscale(images)
        numpy.testing.assert_allclose(grayscale_images, [[[[0], [254], [127]]]])
        self.assertEqual(grayscale_images.shape[3], 1)

    def test_normalise_image_zero_mean_should_result_between_1_to_mins_1(self):
        images = np.array([[1, 1, 1], [255, 255, 255], [128, 128, 128]])
        result = _normalise_image_zero_mean(images)
        print(result)
        numpy.testing.assert_allclose(result, [[-0.99, -0.99, -0.99], [0.99, 0.99, 0.99], [0., 0., 0.]], rtol=0.09)

        # Grayscale Image
        images = np.array([[1], [255], [128]])
        numpy.testing.assert_allclose(_normalise_image_zero_mean(images),
                                      [[-0.99], [0.99], [0.]], rtol=0.09)

    def test_normalise_image_should_result_between_dot5_to_mins_dot5(self):
        images = np.array([[1, 1, 1], [255, 255, 255], [128, 128, 128]])
        result = _normalise_image(images)
        print(result)
        numpy.testing.assert_allclose(result, [[-0.49, -0.49, -0.49], [0.49, 0.49, 0.49], [0., 0., 0.]], rtol=0.1)

        # Grayscale Image
        images = np.array([[1], [255], [128]])
        numpy.testing.assert_allclose(_normalise_image(images),
                                      [[-0.49], [0.49], [0.]], rtol=0.09)

    def test_normalise_image_whitening(self):
        images = np.array([[[[1, 1, 1], [255, 255, 255], [128, 128, 128]]]])
        result = _normalise_image_whitening(images)
        print(result)
        numpy.testing.assert_allclose(result, [[[[-1.22474492, -1.22474492, -1.22474492],
                                                 [1.22474492,  1.22474492,  1.22474492],
                                                 [0.,          0.,          0.]]]])

    def test_enhance_one_image_with_tensorflow_random_operations(self):
        image = real_data_provider_no_shuffer.X_train[0:1]
        result = _enhance_one_image_with_tensorflow_random_operations(image[0], 30)
        self.assertTrue(result.shape, (30, 32, 32, 3))
        result = numpy.append(result, image, axis=0)
        TrainingPlotter.combine_images(result, "./test_data_enhancement/random_generate_with_tensorflow.png")

    def test_enhance_with_tensorflow_random_bulk(self):
        original_images = real_data_provider_no_shuffer.X_train[0:2]
        result_images, _ = enhance_with_tensorflow_brightness_contrast_bulk(original_images,
                                                                         real_data_provider_no_shuffer.y_train[0:2], 3)
        TrainingPlotter.combine_images(result_images, "./test_data_enhancement/random_generate_with_tensorflow_bulk.png")
