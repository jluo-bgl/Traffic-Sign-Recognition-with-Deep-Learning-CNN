import unittest
from pandas import DataFrame
import pandas as pd
from .data_explorer import SignNames
from .data_explorer import DataExplorer
from .traffic_data import TrafficDataSets
from .data_explorer import TrainingPlotter
from tensorflow.python.framework import dtypes
import pickle
import os


class TestSignNames(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.sign_names = SignNames("signnames.csv")
        self.data_frame = self.sign_names.data_frame

    def test_init(self):
        data_frame = self.sign_names.data_frame
        self.assertEqual(type(data_frame), DataFrame)
        self.assertEqual(len(data_frame), 43, "There are should 43 signs")
        self.assertTrue(data_frame.index.is_unique, "ClassId should unique")

    def test_sign_name_by_id(self):
        self.assertEqual(self.sign_names.sign_name_by_id(14), "Stop")


class TestDataExplorer(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        training_file = "train.p"
        testing_file = "test.p"
        with open(training_file, mode='rb') as f:
            train = pickle.load(f)

        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)

        if not os.path.exists("./explorer"):
            os.makedirs("./explorer")

        self.X_train = train['features']
        self.y_train = train['labels']
        self.X_test = test['features']
        self.y_test = test['labels']
        self.sign_names = SignNames("signnames.csv")

        self.traffic_datasets = TrafficDataSets(training_file, testing_file,
                                                dtype=dtypes.uint8, grayscale=False, one_hot_encode=False)
        datasets = self.traffic_datasets
        self.explorer = DataExplorer(self.sign_names, datasets.train.images, datasets.train.labels,
                                datasets.validation.images, datasets.validation.labels,
                                datasets.test.images, datasets.test.labels)

    def test_init(self):
        self.assertEqual(len(self.explorer.Test_Features), 12630, "Pictures of Test Features should be 12,630")
        self.assertTupleEqual(self.explorer.Test_Features.shape, (12630, 32, 32, 3))
        self.assertEqual(len(self.X_train), 39209, "Pictures of Test Features should be 39209")
        self.assertTupleEqual(self.explorer.Train_Features.shape, (39209, 32, 32, 3))
        self.assertTupleEqual(self.explorer.train_labels.shape, (39209, ))
        self.assertTupleEqual(self.explorer.test_labels.shape, (12630, ))

    def test_sample(self):
        plt = DataExplorer._sample(self.X_train, self.y_train, slice(1, 3), self.sign_names)
        plt.savefig('./explorer/test_sample.png')
        # TODO How to check the titles?
        self.assertTupleEqual(tuple(plt.gca().figure.bbox.size), (240., 320.))
        self.assertTrue(plt is not None)

    def test_sample_training_data(self):
        self.explorer.sample_training_data(slice(0, 10)).savefig("./explorer/training_sample_1_10.png")
        self.explorer.sample_training_data(slice(10000, 10010)).savefig("./explorer/training_sample_end_of_speed.png")
        self.explorer.sample_training_data(slice(20000, 20010)).savefig("./explorer/training_sample_priority_road.png")

    def test_sample_testing_data(self):
        # During the data understanding part, obersaved that below line produced wrong picture(data)?
        # for i in range(1000)
        self.explorer.sample_testing_data(slice(1000, 1040)).savefig("./explorer/testing_sample_end_of_speed.png")
        # ValueError: left cannot be >= right
        # self.explorer.sample_testing_data(slice(1040, 1080)).savefig("./explorer/testing_sample_1.png")
        # self.explorer.sample_testing_data(slice(2000, 2010)).savefig("./explorer/testing_sample_priority_road.png")

    def test_data_distribution(self):
        distribution = self.explorer._data_distribution(self.y_test, self.sign_names)
        self.assertEqual(len(distribution), 43)

    def test_training_data_distribution(self):
        self.explorer.bar_chart_data_distribution(self.explorer.training_data_distribution(), "Training Data Distribution")\
            .savefig("./explorer/training_data_distribution.png")

    def test_testing_data_distribution(self):
        self.explorer.bar_chart_data_distribution(self.explorer.testing_data_distribution(), "Testing Data Distribution")\
            .savefig("./explorer/testing_data_distribution.png")


class TestTrainingPlotter(unittest.TestCase):
    def test_plot_confusion_matrix_text(self):
        y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
        y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
        plt = TrainingPlotter.plot_confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
        plt.savefig('./explorer/confusion_matrix_test_plot_confusion_matrix_text.png')

    def test_plot_confusion_matrix_float(self):
        y_true = [2, 0, 2, 2, 0, 1]
        y_pred = [0, 0, 2, 2, 0, 2]
        plt = TrainingPlotter.plot_confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        plt.savefig('./explorer/confusion_matrix_test_plot_confusion_matrix_float.png')

    def test_loss_acc_plot(self):
        plotter = TrainingPlotter("the title", './explorer/test_loss_acc_plot.png', show_plot_window=False)
        for epoch in range(50):
            loss_train, acc_train = 100 - epoch, epoch
            if epoch % 10 == 0:
                loss_val, acc_val = 100 - epoch - 3, epoch - 5
            else:
                loss_val, acc_val = None, None
            plotter.add_loss_accuracy_to_plot(epoch, loss_train, acc_train, loss_val, acc_val, redraw=True)

        plotter.safe_shut_down()