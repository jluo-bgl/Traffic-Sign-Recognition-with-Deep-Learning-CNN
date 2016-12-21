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
from tensorflow.contrib.layers import flatten
from traffic.traffic_data import TrafficDataSets
import logging
import logging.config

logging.config.fileConfig('logging.conf')

logging.info("Test")

EPOCHS = 10
BATCH_SIZE = 50

LABEL_SIZE = TrafficDataSets.NUMBER_OF_CLASSES


# LeNet architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
#
# Don't worry about anything else in the file too much, all you have to do is
# create the LeNet and return the result of the last fully connected layer.
def LeNet(x):
    # Reshape from 2D to 4D. This prepares the data for
    # convolutional and pooling layers.
    # x = tf.reshape(x, (-1, 28, 28, 1))
    # Pad 0s to 32x32. Centers the digit further.
    # Add 2 rows/columns on each side for height and width dimensions.
    # x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")

    # 28x28x6
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6)))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    conv1 = tf.nn.relu(conv1)

    # 14x14x6
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 10x10x16
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16)))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    conv2 = tf.nn.relu(conv2)

    # 5x5x16
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten
    fc1 = flatten(conv2)
    # (5 * 5 * 16, 120)
    fc1_shape = (fc1.get_shape().as_list()[-1], 120)

    fc1_W = tf.Variable(tf.truncated_normal(shape=(fc1_shape)))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, LABEL_SIZE)))
    fc2_b = tf.Variable(tf.zeros(LABEL_SIZE))
    return tf.matmul(fc1, fc2_W) + fc2_b


# MNIST consists of 28x28x1, grayscale images
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
# Classify over 10 digits 0-9
y = tf.placeholder(tf.float32, (None, LABEL_SIZE))
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
    traffic_datas = TrafficDataSets('train.p', 'test.p')

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        steps_per_epoch = traffic_datas.train.num_examples // BATCH_SIZE

        # Train model
        for i in range(EPOCHS):
            for step in range(steps_per_epoch):
                batch_x, batch_y = traffic_datas.train.next_batch(BATCH_SIZE)
                loss = sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

            val_loss, val_acc = eval_data(traffic_datas.validation)
            logging.info("EPOCH {} ...".format(i + 1))
            logging.info("Validation loss = {:.3f}".format(val_loss))
            logging.info("Validation accuracy = {:.3f}".format(val_acc))

        # Evaluate on the test data
        test_loss, test_acc = eval_data(traffic_datas.test)
        logging.info("Test loss = {:.3f}".format(test_loss))
        logging.info("Test accuracy = {:.3f}".format(test_acc))

"""
2016-12-20 16:37:13,743 - EPOCH 1 ...
2016-12-20 16:37:13,744 - Validation loss = 17.880
2016-12-20 16:37:13,744 - Validation accuracy = 0.073
2016-12-20 16:37:52,670 - EPOCH 2 ...
2016-12-20 16:37:52,670 - Validation loss = 6.292
2016-12-20 16:37:52,670 - Validation accuracy = 0.060
2016-12-20 16:38:32,021 - EPOCH 3 ...
2016-12-20 16:38:32,021 - Validation loss = 4.410
2016-12-20 16:38:32,021 - Validation accuracy = 0.059
2016-12-20 16:39:10,526 - EPOCH 4 ...
2016-12-20 16:39:10,526 - Validation loss = 3.929
2016-12-20 16:39:10,526 - Validation accuracy = 0.060
2016-12-20 16:39:50,042 - EPOCH 5 ...
2016-12-20 16:39:50,042 - Validation loss = 3.730
2016-12-20 16:39:50,042 - Validation accuracy = 0.061
2016-12-20 16:40:28,760 - EPOCH 6 ...
2016-12-20 16:40:28,760 - Validation loss = 3.625
2016-12-20 16:40:28,760 - Validation accuracy = 0.060
2016-12-20 16:41:09,799 - EPOCH 7 ...
2016-12-20 16:41:09,800 - Validation loss = 3.568
2016-12-20 16:41:09,800 - Validation accuracy = 0.065
2016-12-20 16:41:49,696 - EPOCH 8 ...
2016-12-20 16:41:49,696 - Validation loss = 3.532
2016-12-20 16:41:49,696 - Validation accuracy = 0.059
2016-12-20 16:42:30,373 - EPOCH 9 ...
2016-12-20 16:42:30,373 - Validation loss = 3.518
2016-12-20 16:42:30,373 - Validation accuracy = 0.059
2016-12-20 16:43:10,400 - EPOCH 10 ...
2016-12-20 16:43:10,400 - Validation loss = 3.511
2016-12-20 16:43:10,400 - Validation accuracy = 0.058
2016-12-20 16:43:14,862 - Test loss = 3.712
2016-12-20 16:43:14,862 - Test accuracy = 0.061
"""