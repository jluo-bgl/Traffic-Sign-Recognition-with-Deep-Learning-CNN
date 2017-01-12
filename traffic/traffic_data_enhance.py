import numpy
import tensorflow as tf
import scipy.ndimage
import scipy.misc
from tensorflow.python.framework import dtypes
from .traffic_data import TrafficDataProvider
import math


"""
    Normally you should enhance feature size first, then grayscale, then normalise
"""


def normalise_image_positive(provider):
    return apply_func_to_images(provider, _normalise_image_positive)


def normalise_images(provider):
    return apply_func_to_images(provider, _normalise_image)


def normalise_image_zero_mean(provider):
    return apply_func_to_images(provider, _normalise_image_zero_mean)


def grayscale(provider):
    return apply_func_to_images(provider, _image_grayscale)


def normalise_image_unit_variance(provider):
    return apply_func_to_images(provider, _normalise_image_unit_variance)


def normalise_image_whitening(provider):
    return apply_func_to_images(provider, _normalise_image_whitening)


def apply_func_to_images(provider, func):
    return TrafficDataProvider(
        X_train_array=func(provider.X_train),
        y_train_array=provider.y_train,
        X_validation_array=func(provider.X_validation),
        y_validation_array=provider.y_validation,
        X_test_array=func(provider.X_test),
        y_test_array=provider.y_test
    )


def _normalise_image_whitening(images):
    tensor = tf.map_fn(lambda image: tf.image.per_image_standardization(image), images, dtype=dtypes.float32)
    return tf.Session().run(tensor)


def _normalise_image_positive(images):
    """
    normalise image to 0.0 to 1.0
    :param images:
    :return:
    """
    images = numpy.multiply(images, 1.0 / 255.0)
    return images

def _normalise_image(images):
    """
    normalise image to 0.0 to 1.0
    :param images:
    :return:
    """
    # Convert from [0, 255] -> [0.0, 1.0].
    images = images - 128
    images = numpy.multiply(images, 1.0 / 255.0)
    return images


def _normalise_image_unit_variance(images):
    """
    normalise image to 0.0 to 1.0
    :param images:
    :return:
    """
    images = (images - images.mean(axis=0)) / images.std(axis=0)
    return images


def _normalise_image_zero_mean(images):
    images = images - 128
    images = images / 128
    return images


def _image_grayscale(images):
    images = tf.image.rgb_to_grayscale(images)
    images = tf.Session().run(images)
    return images


def enhance_with_brightness_contrast(images, labels, ratio):
    """
    Please Note, this method currently too slow
    :param images:
    :param labels:
    :param ratio:
    :return:
    """
    return enhance_with_function(images, labels, ratio, _enhance_one_image_with_tensorflow_random_operations)


def _enhance_images_with_tensorflow_random_operations(images):
    def generator_one(image):
        distorted_image = image
        # distorted_image = tf.random_crop(image, [32, 32, 3])
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=0.5)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)
        return distorted_image

    tensor = tf.map_fn(
        lambda image: generator_one(image),
        images,
        dtype=dtypes.uint8)
    return tf.Session().run(tensor)


def enhance_with_tensorflow_brightness_contrast_bulk(images, labels, ratio):
    result = numpy.array(images)
    result_lables = numpy.array(labels)
    for index in range(ratio):
        result = numpy.append(result, _enhance_images_with_tensorflow_random_operations(images), axis=0)
        result_lables = numpy.append(result_lables, labels, axis=0)
    return result, result_lables


def enhance_with_random_rotate(images, labels, ratio):
    return enhance_with_function(images, labels, ratio, _enhance_one_image_with_rotate_randomly)


def enhance_with_random_zoomin(images, labels, ratio):
    """
    :param images:
    :param labels:
    :param ratio: the ratio of max input class. for example, highest sample count is 1000, ratio is 3, the result
    will be around 1000 * 3 * how_many_classes
    :return: new genrated features and labels
    """
    return enhance_with_function(images, labels, ratio, _enhance_one_image_with_zoomin_randomly)


def enhance_with_random_zoomin_and_rotate(images, labels, ratio):
    """
    :param images:
    :param labels:
    :param ratio: the ratio of max input class. for example, highest sample count is 1000, ratio is 3, the result
    will be around 1000 * 3 * how_many_classes
    :return: new genrated features and labels
    """
    return enhance_with_function(
        images, labels, ratio,
        _enhance_one_image_with_random_funcs(
            [
                _enhance_one_image_with_rotate_randomly,
                _enhance_one_image_with_zoomin_randomly
            ]
        ))


def enhance_with_function(images, labels, ratio, enhance_func):
    """
    :param images:
    :param labels:
    :param ratio: the ratio of max input class. for example, highest sample count is 1000, ratio is 3, the result
    will be around 1000 * 3 * how_many_classes
    :param enhance_func the func used for enhance f(image, label, how_many_to_generate)
    :return: new genrated features and labels
    """
    inputs_per_class = numpy.bincount(labels)
    max_inputs = numpy.max(inputs_per_class)

    # One Class
    for i in range(len(inputs_per_class)):
        input_ratio = math.ceil((max_inputs * ratio - inputs_per_class[i]) / inputs_per_class[i])
        print("generating class:{} with ratio:{}, max input:{}, current:{}".format(
            i, input_ratio, max_inputs, inputs_per_class[i]))

        if input_ratio <= 1:
            continue

        new_features = []
        new_labels = []
        mask = numpy.where(labels == i)

        for feature in images[mask]:
            generated_images = enhance_func(feature, input_ratio)
            for generated_image in generated_images:
                new_features.append(generated_image)
                new_labels.append(i)

        images = numpy.append(images, new_features, axis=0)
        labels = numpy.append(labels, new_labels, axis=0)

    return images, labels


def _flatten(listoflists):
    return [item for list in listoflists for item in list]


_IMAGE_SCALES = numpy.arange(0.9, 1.1, 0.02)
_IMAGE_CUT_RATIOS = numpy.arange(0.05, 0.2, 0.02)

def _zoomin_image_randomly(image):
    """
    resize image randomly between 0.9 and 1.1 but keep output still same
    :param image: the image to resize
    :return: image resized randomly between 0.9 and 1.1
    """
    scale = numpy.random.choice(_IMAGE_CUT_RATIOS)
    lx, ly, _ = image.shape
    first_run = image[int(lx * scale): - int(lx * scale), int(ly * scale): - int(ly * scale), :]
    return scipy.misc.imresize(first_run, (32, 32))


def _enhance_one_image_with_zoomin_randomly(image, how_many_to_generate):
    generated_images = []
    for index in range(how_many_to_generate):
        generated_image = _zoomin_image_randomly(image)
        generated_images.append(generated_image)

    return generated_images


_IMAGE_ROTATE_ANGLES = numpy.arange(-20, 20, 3)


def _enhance_one_image_with_rotate_randomly(image, how_many_to_generate):
    generated_images = []
    for index in range(how_many_to_generate):
        generated_images.append(
            scipy.ndimage.rotate(image,
                                 numpy.random.choice(_IMAGE_ROTATE_ANGLES),
                                 reshape=False))

    return generated_images


def _enhance_one_image_with_tensorflow_random_operations(image, how_many_to_generate):
    def process_stack():
        distorted_image = image
        # distorted_image = tf.random_crop(image, [32, 32, 3])
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=0.5)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)
        return distorted_image

    tensor = tf.map_fn(lambda index: process_stack(), numpy.array(list(range(how_many_to_generate))), dtype=dtypes.uint8)
    return tf.Session().run(tensor)


def _enhance_one_image_with_random_funcs(enhance_funcs):
    def __f(image, how_many_to_generate):
        func_indeies = numpy.random.randint(0, len(enhance_funcs), size=how_many_to_generate)
        return _flatten(map(lambda i: enhance_funcs[i](image, 1), func_indeies))

    return __f


def _enhance_one_image_randomly(image, label, how_many_to_generate):
    """
    Didn't make this working. the color channel seems been changed by ImageDataGenerator
    :param image:
    :param label:
    :param how_many_to_generate:
    :return:
    """
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=0.,
        width_shift_range=0.,
        height_shift_range=0.,
        featurewise_center=False,
        featurewise_std_normalization=False,
        rescale=None,
        shear_range=0.,
        zoom_range=0.,
        horizontal_flip=False,
        vertical_flip=False,
        channel_shift_range=0.,
        fill_mode='nearest',
        dim_ordering='tf')
    iterator = datagen.flow(numpy.array([image]), numpy.array([label]), batch_size=how_many_to_generate,
                                      shuffle=False, seed=None)
    generated_images, generated_labels = [], []
    for index in range(how_many_to_generate):
        generated_image, generated_label = iterator.next()
        for item in generated_image:
            generated_images.append(item)
        for item in generated_label:
            generated_labels.append(item)

    return generated_images, generated_labels
