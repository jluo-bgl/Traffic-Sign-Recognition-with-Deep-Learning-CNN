import unittest
from pandas import DataFrame
import pandas as pd
from .data_explorer import SignNames
from .data_explorer import DataExplorer
from .traffic_data import TrafficDataSets
from .traffic_data import TrafficDataProvider
from .traffic_data import TrafficDataRealFileProvider
from .traffic_data import DataSetWithGenerator
from .data_explorer import TrainingPlotter
from tensorflow.python.framework import dtypes
import pickle
import os

real_data_provider = TrafficDataRealFileProvider(split_validation_from_train=True)
real_data_provider_no_shuffer = TrafficDataRealFileProvider(split_validation_from_train=False)
clear_data_range = slice(80, 90)
clear_subset_data_provider = TrafficDataProvider(
    real_data_provider_no_shuffer.X_train[clear_data_range],
    real_data_provider_no_shuffer.y_train[clear_data_range],
    real_data_provider_no_shuffer.X_test[clear_data_range],
    real_data_provider_no_shuffer.y_test[clear_data_range],
    split_validation_from_train=False
)

class TestTrafficDataSets(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.training_file = "train.p"
        self.testing_file = "test.p"
        self.traffic_datasets = TrafficDataSets(real_data_provider,
                                                dtype = dtypes.uint8, grayscale = False, one_hot_encode = False)

    def test_init(self):
        self.assertTrue(self.traffic_datasets.test is not None)
        self.assertTrue(self.traffic_datasets.validation is not None)
        self.assertTrue(self.traffic_datasets.train is not None)
        self.assertEqual(self.traffic_datasets.NUMBER_OF_CLASSES, 43, "we have 43 kind of traffic signs")
        self.assertEqual(self.traffic_datasets.train.num_examples, 27446, "70% of training data")
        self.assertEqual(self.traffic_datasets.validation.num_examples, 11763, "30% of validation data")
        self.assertEqual(self.traffic_datasets.test.num_examples, 12630, "30% of test data")

    def test_gray_scale_should_contains_one_chanel(self):
        traffic_datasets = TrafficDataSets(
            TrafficDataRealFileProvider(split_validation_from_train=True), dtype=dtypes.float32, grayscale=True)
        self.assertEqual(traffic_datasets.test.images.shape[3], 1)
        self.assertTrue(traffic_datasets.test.is_grayscale)

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
        datagen = DataSetWithGenerator._data_generator_factory()
        x = clear_subset_data_provider.X_train
        y = clear_subset_data_provider.y_train
        DataExplorer._sample(x, y, slice(0, len(y)), SignNames("signnames.csv")).savefig(test_image_folder + "/original.png")
        i = 0
        for batch in datagen.flow(x, y, batch_size=6,
                                  save_to_dir=test_image_folder, save_prefix='traffic', save_format='jpeg'):
            i += 1
            if i > 20:
                break  # otherwise the generator would loop indefinitely

    def test_next_batch(self):
        test_image_folder = get_and_make_sure_folder_exists("./test_next_batch_with_generator")
        x = clear_subset_data_provider.X_train
        y = clear_subset_data_provider.y_train
        DataExplorer._sample(x, y, slice(0, len(y)), SignNames("signnames.csv")).savefig(
            test_image_folder + "/original.png")
        dataset = DataSetWithGenerator(x, y, dtype=dtypes.uint8, grayscale=False)
        i = 0
        for batch in dataset.next_batch(batch_size=5, save_to_dir=test_image_folder, save_prefix='test_next_batch_generator'):
            i += 1
            if i > 20:
                break

