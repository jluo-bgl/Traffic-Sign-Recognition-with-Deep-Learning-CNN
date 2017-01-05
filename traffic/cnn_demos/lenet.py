"""
LeNet Architecture

HINTS for layers:

    Convolutional layers:

    tf.nn.conv2d
    tf.nn.max_pool

    For preparing the convolutional layer output for the
    fully connected layers.

    tf.contrib.flatten
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten
import logging
import logging.config

logging.config.fileConfig('logging.conf')

logging.info("Test")

EPOCHS = 10
BATCH_SIZE = 150


# LeNet architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
#
# Don't worry about anything else in the file too much, all you have to do is
# create the LeNet and return the result of the last fully connected layer.
def LeNet(x):
    # Reshape from 2D to 4D. This prepares the data for
    # convolutional and pooling layers.
    x = tf.reshape(x, (-1, 28, 28, 1))
    # Pad 0s to 32x32. Centers the digit further.
    # Add 2 rows/columns on each side for height and width dimensions.
    x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")

    # 28x28x6
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), stddev=0.1))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    conv1 = tf.nn.relu(conv1)

    # 14x14x6
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 10x10x16
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), stddev=0.1))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    conv2 = tf.nn.relu(conv2)

    # 5x5x16
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten
    fc1 = flatten(conv2)
    # (5 * 5 * 16, 120)
    fc1_shape = (fc1.get_shape().as_list()[-1], 120)

    fc1_W = tf.Variable(tf.truncated_normal(shape=(fc1_shape), stddev=0.1))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 10), stddev=0.1))
    fc2_b = tf.Variable(tf.zeros(10))
    return tf.matmul(fc1, fc2_W) + fc2_b


# MNIST consists of 28x28x1, grayscale images
x = tf.placeholder(tf.float32, (None, 28, 28, 1))
# Classify over 10 digits 0-9
y = tf.placeholder(tf.float32, (None, 10))
fc2 = LeNet(x)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2, y))
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def eval_data(dataset):
    """
    Given a dataset as input returns the loss and accuracy.
    """
    # If dataset.num_examples is not divisible by BATCH_SIZE
    # the remainder will be discarded.
    # Ex: If BATCH_SIZE is 64 and training set has 55000 examples
    # steps_per_epoch = 55000 // 64 = 859
    # num_examples = 859 * 64 = 54976
    #
    # So in that case we go over 54976 examples instead of 55000.
    steps_per_epoch = dataset.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc, total_loss = 0, 0
    sess = tf.get_default_session()
    for step in range(steps_per_epoch):
        batch_x, batch_y = dataset.next_batch(BATCH_SIZE)
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_x, y: batch_y})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])
    return total_loss / num_examples, total_acc / num_examples


if __name__ == '__main__':
    # Load data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        steps_per_epoch = mnist.train.num_examples // BATCH_SIZE

        # Train model
        for i in range(EPOCHS):
            for step in range(steps_per_epoch):
                batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
                loss = sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

            val_loss, val_acc = eval_data(mnist.validation)
            logging.info("EPOCH {} ...".format(i + 1))
            logging.info("Validation loss = {:.3f}".format(val_loss))
            logging.info("Validation accuracy = {:.3f}".format(val_acc))

        # Evaluate on the test data
        test_loss, test_acc = eval_data(mnist.test)
        logging.info("Test loss = {:.3f}".format(test_loss))
        logging.info("Test accuracy = {:.3f}".format(test_acc))

"""
2016-12-19 17:47:24,345 - root - INFO - EPOCH 1 ...
2016-12-19 17:47:24,345 - root - INFO - Validation loss = 67.183
2016-12-19 17:47:24,346 - root - INFO - Validation accuracy = 0.850
2016-12-19 17:48:17,360 - root - INFO - EPOCH 2 ...
2016-12-19 17:48:17,360 - root - INFO - Validation loss = 29.010
2016-12-19 17:48:17,360 - root - INFO - Validation accuracy = 0.918
2016-12-19 17:48:57,724 - root - INFO - EPOCH 3 ...
2016-12-19 17:48:57,724 - root - INFO - Validation loss = 20.797
2016-12-19 17:48:57,724 - root - INFO - Validation accuracy = 0.928
2016-12-19 17:49:42,810 - root - INFO - EPOCH 4 ...
2016-12-19 17:49:42,810 - root - INFO - Validation loss = 14.454
2016-12-19 17:49:42,810 - root - INFO - Validation accuracy = 0.939
2016-12-19 17:50:23,963 - root - INFO - EPOCH 5 ...
2016-12-19 17:50:23,963 - root - INFO - Validation loss = 11.705
2016-12-19 17:50:23,963 - root - INFO - Validation accuracy = 0.944
2016-12-19 17:51:04,278 - root - INFO - EPOCH 6 ...
2016-12-19 17:51:04,279 - root - INFO - Validation loss = 9.270
2016-12-19 17:51:04,279 - root - INFO - Validation accuracy = 0.948
2016-12-19 17:51:46,246 - root - INFO - EPOCH 7 ...
2016-12-19 17:51:46,247 - root - INFO - Validation loss = 7.296
2016-12-19 17:51:46,247 - root - INFO - Validation accuracy = 0.952
2016-12-19 17:52:28,131 - root - INFO - EPOCH 8 ...
2016-12-19 17:52:28,131 - root - INFO - Validation loss = 6.229
2016-12-19 17:52:28,131 - root - INFO - Validation accuracy = 0.958
2016-12-19 17:53:12,466 - root - INFO - EPOCH 9 ...
2016-12-19 17:53:12,466 - root - INFO - Validation loss = 5.384
2016-12-19 17:53:12,466 - root - INFO - Validation accuracy = 0.958
2016-12-19 17:53:56,107 - root - INFO - EPOCH 10 ...
2016-12-19 17:53:56,107 - root - INFO - Validation loss = 4.722
2016-12-19 17:53:56,107 - root - INFO - Validation accuracy = 0.962
2016-12-19 17:53:58,941 - root - INFO - Test loss = 4.481
2016-12-19 17:53:58,941 - root - INFO - Test accuracy = 0.965
"""