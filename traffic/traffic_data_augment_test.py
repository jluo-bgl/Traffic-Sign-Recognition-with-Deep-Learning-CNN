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


class TestTrafficDataGenerator(unittest.TestCase):

    @staticmethod
    def generate(ratio):
        images, labels = enhance_with_brightness_contrast(real_data_provider.X_train, real_data_provider.y_train, ratio)
        provider = real_data_provider.to_other_provider(images, labels)
        provider.save_to_file("traffic_data_training_{}".format(len(images)))

    def test_generate_brightness_contrast_data(self):
        self.generate(1)

    def test_generate_brightness_contrast_data_2(self):
        self.generate(2)
