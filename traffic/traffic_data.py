import numpy
import pickle
from tensorflow.python.framework import dtypes
from sklearn.cross_validation import train_test_split
import tensorflow as tf
from enum import Enum
import scipy.ndimage
import scipy.misc


class DataSetType(Enum):
    Training = 1
    TestAndValudation = 2

class TrafficDataProvider(object):
    def __init__(self, X_train_array, y_train_array, X_test_array, y_test_array, split_validation_from_train=False):


        X_train, X_validation, y_train, y_validation = None, None, None, None
        if split_validation_from_train:
            X_train, X_validation, y_train, y_validation = train_test_split(X_train_array, y_train_array,
                                                                            test_size=0.10, random_state=42)
        else:
            X_train, y_train = X_train_array, y_train_array
            X_validation = X_train_array[0:1000]
            y_validation = y_train_array[0:1000]

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


class TrafficDataEnhancement(object):

    IMAGE_SCALES = numpy.arange(0.9, 1.1, 0.02)

    @staticmethod
    def resize_image_randomly(image):
        """
        resize image randomly between 0.9 and 1.1 but keep output still same
        :param image: the image to resize
        :return: image resized randomly between 0.9 and 1.1
        """
        scale = numpy.random.choice(TrafficDataEnhancement.IMAGE_SCALES)
        first_run = scipy.misc.imresize(image, scale)
        return scipy.misc.imresize(first_run, (32, 32))

    @staticmethod
    def enhance(train_features, train_labels):
        inputs_per_class = numpy.bincount(train_labels)
        max_inputs = numpy.max(inputs_per_class)
        angles = [-5, 5, -10, 10, -15, 15, -20, 20, -3, 3, -6, 6, -8, 8, -12, 12, -14, 14]

        for i in range(len(inputs_per_class)):
            input_ratio = min(int(max_inputs / inputs_per_class[i]) - 1, len(angles) - 1)

            if input_ratio <= 1:
                continue

            new_features = []
            new_labels = []
            mask = numpy.where(train_labels == i)

            for j in range(input_ratio):
                for feature in train_features[mask]:
                    new_features.append(scipy.ndimage.rotate(feature, angles[j], reshape=False))
                    new_labels.append(i)

            train_features = numpy.append(train_features, new_features, axis=0)
            train_labels = numpy.append(train_labels, new_labels, axis=0)

        return train_features, train_labels

    def enhance_one_class(self, X_train_one_class, y_label_one_class, how_many_to_add):
        pass


class TrafficDataSets(object):
    NUMBER_OF_CLASSES = 43

    def __init__(self, data_provider, dtype=dtypes.float32, grayscale = False, one_hot_encode = True,
                 training_dataset_factory=lambda X, y, dtype, grayscale: DataSet(X, y, 500, dtype, grayscale),
                 test_dataset_factory=lambda X, y, dtype, grayscale: DataSet(X, y, 500, dtype, grayscale)):
        y_train, y_validation, y_test = data_provider.y_train, data_provider.y_validation, data_provider.y_test
        if one_hot_encode:
            y_train, y_validation, y_test = dense_to_one_hot(data_provider.y_train, TrafficDataSets.NUMBER_OF_CLASSES), \
                                            dense_to_one_hot(data_provider.y_validation, TrafficDataSets.NUMBER_OF_CLASSES), \
                                            dense_to_one_hot(data_provider.y_test, TrafficDataSets.NUMBER_OF_CLASSES)

        self.train = training_dataset_factory(data_provider.X_train, y_train, dtype, grayscale)
        self.validation = test_dataset_factory(data_provider.X_validation, y_validation, dtype, grayscale)
        self.test = test_dataset_factory(data_provider.X_test, y_test, dtype, grayscale)


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
                 batch_size,
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

        self.batch_size = batch_size

        images = images.astype(numpy.float32)
        if dtype == dtypes.float32:
            images = DataSet.normalise_image(images)

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

    @staticmethod
    def normalise_image(images):
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images - 127
        images = numpy.multiply(images, 1.0 / 255.0)
        return images

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

    def reset(self):
        pass

    def next_batch(self):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += self.batch_size
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
            self._index_in_epoch = self.batch_size
            assert self.batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


class DataSetWithGenerator(DataSet):
    def __init__(self,
                 images,
                 labels,
                 batch_size,
                 dataset_type,
                 dtype=dtypes.float32,
                 grayscale=False,
                 save_to_dir=None, save_prefix=None):
        super().__init__(images, labels, batch_size, dtype, grayscale)
        if DataSetType.Training == dataset_type:
            self.datagen = DataSetWithGenerator._training_data_generator_factory()
        else:
            self.datagen = DataSetWithGenerator._test_data_generator_factory()

        self.datagen.fit(self._images)
        self.iterator = self.datagen.flow(self._images, self._labels, batch_size=batch_size,
                                         shuffle=True, seed=1234,
                                         save_to_dir=save_to_dir, save_prefix=save_prefix, save_format='jpeg')

    @staticmethod
    def _training_data_generator_factory():
        from keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            featurewise_center=True,
            featurewise_std_normalization=False,
            rescale=1. / 255,
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
            featurewise_center=True,
            featurewise_std_normalization=False,
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode='nearest',
            dim_ordering='tf')
        return datagen

    def reset(self):
        self.iterator.reset()

    def next_batch(self):
        images, y = self.iterator.next()
        images = DataSet.normalise_image(images)
        return images, y

