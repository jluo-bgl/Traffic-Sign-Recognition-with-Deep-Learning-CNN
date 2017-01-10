import numpy
import pickle
from tensorflow.python.framework import dtypes
from sklearn.model_selection import train_test_split
import tensorflow as tf
from enum import Enum
import scipy.ndimage
import scipy.misc
from sklearn.utils import shuffle


class TrafficDataProvider(object):
    """
    provide data to neural network
    """
    def __init__(self,
                 X_train_array, y_train_array, X_validation_array, y_validation_array, X_test_array, y_test_array):
        self.X_train = X_train_array
        self.X_validation = X_validation_array
        self.y_train = y_train_array
        self.y_validation = y_validation_array
        self.X_test = X_test_array
        self.y_test = y_test_array

    def to_other_provider(self, X_train_overwrite=None, y_train_overwrite=None):
        if X_train_overwrite is not None:
            X_train = X_train_overwrite
        else:
            X_train = self.X_train
        if y_train_overwrite is not None:
            y_train = y_train_overwrite
        else:
            y_train = self.y_train
        return TrafficDataProvider(X_train, y_train, self.X_validation, self.y_validation, self.X_test, self.y_test)

    def save_to_file(self, file_name):
        data = {
            "train_features": self.X_train,
            "train_labels": self.y_train,
            "validation_features": self.X_validation,
            "validation_labels": self.y_validation,
            "test_features": self.X_test,
            "test_labels": self.y_test
        }
        with open(file_name, mode='wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_from_file(file_name):
        with open(file_name, mode='rb') as f:
            data = pickle.load(f)

        return TrafficDataProvider(
            X_train_array=data["train_features"],
            y_train_array=data["train_labels"],
            X_validation_array=data["validation_features"],
            y_validation_array=data["validation_labels"],
            X_test_array=data["test_features"],
            y_test_array=data["test_labels"]
        )

    @classmethod
    def from_other_provider(cls, data_provider):
        return cls(data_provider.X_train, data_provider.y_train,
                   data_provider.X_validation, data_provider.y_validation,
                   data_provider.X_test, data_provider.y_test)


class TrafficDataProviderAutoSplitValidationData(TrafficDataProvider):
    def __init__(self, X_train_array, y_train_array, X_test_array, y_test_array,
                 split_validation_from_train=False, validation_size=0.20):
        """
        Provide X_train and X_test, calculate validation set from X_train.
        :param X_train_array:
        :param y_train_array:
        :param X_test_array:
        :param y_test_array:
        :param split_validation_from_train: if true will shuffle data and split validation based on ratio of
        validation_size, otherwise simple copy 1 to 1000 images from X_train
        :param validation_size: how much to split validation from training set.
        """
        if split_validation_from_train:
            X_train, X_validation, y_train, y_validation = train_test_split(X_train_array, y_train_array,
                                                                            test_size=validation_size, random_state=42)
        else:
            X_train, y_train = X_train_array, y_train_array
            X_validation = X_train_array[0:1000]
            y_validation = y_train_array[0:1000]

        super().__init__(X_train, y_train, X_validation, y_validation, X_test_array, y_test_array)


class TrafficDataRealFileProviderAutoSplitValidationData(TrafficDataProviderAutoSplitValidationData):
    def __init__(self, training_file="train.p", testing_file="test.p",
                 split_validation_from_train=True, validation_size=0.20):
        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)

        super().__init__(train['features'], train['labels'], test['features'], test['labels'],
                         split_validation_from_train, validation_size)


class TrafficDataSets(object):
    NUMBER_OF_CLASSES = 43

    def __init__(self, data_provider, one_hot_encode=True,
                 training_dataset_factory=lambda X, y: DataSet(X, y),
                 test_dataset_factory=lambda X, y: DataSet(X, y)):
        y_train, y_validation, y_test = data_provider.y_train, data_provider.y_validation, data_provider.y_test
        if one_hot_encode:
            y_train, y_validation, y_test = self.dense_to_one_hot(data_provider.y_train, TrafficDataSets.NUMBER_OF_CLASSES), \
                                            self.dense_to_one_hot(data_provider.y_validation, TrafficDataSets.NUMBER_OF_CLASSES), \
                                            self.dense_to_one_hot(data_provider.y_test, TrafficDataSets.NUMBER_OF_CLASSES)

        self.train = training_dataset_factory(data_provider.X_train, y_train)
        self.validation = test_dataset_factory(data_provider.X_validation, y_validation)
        self.test = test_dataset_factory(data_provider.X_test, y_test)

    @staticmethod
    def dense_to_one_hot(labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        return tf.one_hot(labels_dense, num_classes).eval(session=tf.Session())
        # num_labels = labels_dense.shape[0]
        # index_offset = numpy.arange(num_labels) * num_classes
        # labels_one_hot = numpy.zeros((num_labels, num_classes))
        # labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        # return labels_one_hot


class DataSet(object):
    def __init__(self,
                 images,
                 labels):
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def shuffle(self):
        images, labels = shuffle(self._images, self._labels)
        self._images = images
        self._labels = labels

    @property
    def is_grayscale(self):
        return self._images.shape[3] == 1

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


class DataSetType(Enum):
    Training = 1
    TestAndValudation = 2


class DataSetWithGenerator(DataSet):
    """
    Haven't able to train those dataset yet, always get 0.03 accuracy
    """
    def __init__(self,
                 images,
                 labels,
                 dataset_type,
                 save_to_dir=None, save_prefix=None):
        super().__init__(images, labels)
        if DataSetType.Training == dataset_type:
            self.datagen = DataSetWithGenerator._training_data_generator_factory()
        else:
            self.datagen = DataSetWithGenerator._test_data_generator_factory()

        self.datagen.fit(self._images)
        self.iterator = None
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix

    @staticmethod
    def _training_data_generator_factory():
        from keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            featurewise_center=False,
            featurewise_std_normalization=False,
            zca_whitening=False,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode='nearest',
            dim_ordering='tf')
        return datagen

    @staticmethod
    def _test_data_generator_factory():
        from keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(
            featurewise_center=False,
            featurewise_std_normalization=False,
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode='nearest',
            dim_ordering='tf')
        return datagen

    def next_batch(self, batch_size):
        if self.iterator is None:
            self.iterator = self.datagen.flow(self._images, self._labels, batch_size=batch_size,
                              shuffle=True, seed=1234,
                              save_to_dir=self.save_to_dir, save_prefix=self.save_prefix, save_format='jpeg')

        images, y = self.iterator.next()
        return images, y

