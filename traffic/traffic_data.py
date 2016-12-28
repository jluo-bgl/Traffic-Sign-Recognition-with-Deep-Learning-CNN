import numpy
import pickle
from tensorflow.python.framework import dtypes
from sklearn.cross_validation import train_test_split
import tensorflow as tf


class TrafficDataProvider(object):
    def __init__(self, X_train_array, y_train_array, X_test_array, y_test_array, split_validation_from_train=False):


        X_train, X_validation, y_train, y_validation = None, None, None, None
        if split_validation_from_train:
            X_train, X_validation, y_train, y_validation = train_test_split(X_train_array, y_train_array,
                                                                            test_size=0.30, random_state=42)
        else:
            X_train, y_train = X_train_array, y_train_array

        X_test, y_test = X_test_array, y_test_array

        self.X_train = X_train
        self.X_validation = X_validation
        self.y_train = y_train
        self.y_validation = y_validation
        self.X_test = X_test
        self.y_test = y_test


class TrafficDataRealFileProvider(TrafficDataProvider):
    def __init__(self, split_validation_from_train=True):
        training_file = "train.p"
        testing_file = "test.p"
        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)

        super().__init__(train['features'], train['labels'], test['features'], test['labels'],
                         split_validation_from_train)


class TrafficDataSets(object):
    NUMBER_OF_CLASSES = 43

    def __init__(self, data_provider, dtype = dtypes.float32, grayscale = False, one_hot_encode = True,
                 dataset_factory=lambda X, y, dtype, grayscale: DataSet(X, y, dtype, grayscale)):
        y_train, y_validation, y_test = data_provider.y_train, data_provider.y_validation, data_provider.y_test
        if one_hot_encode:
            y_train, y_validation, y_test = dense_to_one_hot(data_provider.y_train, TrafficDataSets.NUMBER_OF_CLASSES), \
                                            dense_to_one_hot(data_provider.y_validation, TrafficDataSets.NUMBER_OF_CLASSES), \
                                            dense_to_one_hot(data_provider.y_test, TrafficDataSets.NUMBER_OF_CLASSES)

        self.train = dataset_factory(data_provider.X_train, y_train, dtype, grayscale)
        self.validation = dataset_factory(data_provider.X_validation, y_validation, dtype, grayscale)
        self.test = dataset_factory(data_provider.X_test, y_test, dtype, grayscale)


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 dtype=dtypes.float32,
                 grayscale=False):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = images - 127

            images = numpy.multiply(images, 1.0 / 255.0)

            # images = tf.image.convert_image_dtype(images, tf.float32)
            # images = tf.Session().run(images)
            # image_tensor = tf.convert_to_tensor(images)
            # with tf.Session():
            #     image_tensor_whitened = tf.image.per_image_whitening(image_tensor)
            #     images = image_tensor_whitened.eval()

        if grayscale and dtype == dtypes.uint8:
            raise TypeError('grayscale have to setup dtype as float32')

        if grayscale:
            images = tf.image.rgb_to_grayscale(images)
            images = tf.Session().run(images)

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


class DataSetWithGenerator(DataSet):
    def __init__(self,
                 images,
                 labels,
                 dtype=dtypes.float32,
                 grayscale=False):
        super().__init__(images, labels, dtype, grayscale)
        self.datagen = DataSetWithGenerator._data_generator_factory()

    @staticmethod
    def _data_generator_factory():
        from keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode='nearest',
            dim_ordering='tf')
        return datagen

    def next_batch(self, batch_size, save_to_dir=None, save_prefix=None):
        return self.datagen.flow(self._images, self._labels, batch_size=batch_size,
                                 shuffle=True,
                                 save_to_dir= save_to_dir,  save_prefix=save_prefix, save_format='jpeg')

