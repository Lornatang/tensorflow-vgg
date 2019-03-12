import tensorflow as tf
import numpy as np


# print layer information
def print_layer(t):
    print(f"{t.op.name} {t.get_shape().as_list()} \n")


# conv layer op
def Conv2D(x, out, kernel_size, stride, name):
    """

    :param x: input tensor.
    :param out: output tensor.
    :param kernel_size: kernel size.
    :param stride: step length.
    :param name: layer name.
    :return: activation.
    """
    input_x = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[kernel_size, kernel_size, input_x, out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(x, kernel, (1, stride, stride, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        print_layer(activation)
        return activation


# define fully connected
def FullyConnected(x, out, name):
    """

    :param x: input tensor.
    :param out: output tensor.
    :param name: layer name.
    :return: activation
    """
    input_x = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[input_x, out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.Variable(tf.constant(0.1, shape=[out], dtype=tf.float32, name='b'))
        activation = tf.nn.relu_layer(x, kernel, biases, name=scope)
        print_layer(activation)
        return activation


# define max pool layer
def MaxPool2D(input_op, kernel_size, stride, name):
    """

    :param input_op: input tensor.
    :param name: layer name
    :param kernel_size: kernel size.
    :param stride: step length.
    :return: tf.nn.max_pool.
    """
    return tf.nn.max_pool(input_op,
                          ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME',
                          name=name)


# VGG neural network
def vgg19(images, keep_prob, classes):
    """

    :param images: input img tensor.
    :param keep_prob: dropout.
    :param classes: classifier classes.
    :return: pred classes.
    """
    conv1_1 = Conv2D(images, 64, kernel_size=3, stride=1, name='conv1_1')
    conv1_2 = Conv2D(conv1_1, 64, kernel_size=3, stride=1, name='conv1_2')
    pool1 = MaxPool2D(conv1_2, kernel_size=2, stride=2, name='max_pool1')

    conv2_1 = Conv2D(pool1, 128, kernel_size=3, stride=1, name='conv2_1')
    conv2_2 = Conv2D(conv2_1, 128, kernel_size=3, stride=1, name='conv2_2')
    pool2 = MaxPool2D(conv2_2, kernel_size=2, stride=2, name='max_pool2')

    conv3_1 = Conv2D(pool2, 256, kernel_size=3, stride=1, name='conv3_1')
    conv3_2 = Conv2D(conv3_1, 256, kernel_size=3, stride=1, name='conv3_2')
    conv3_3 = Conv2D(conv3_2, 256, kernel_size=3, stride=1, name='conv3_3')
    conv3_4 = Conv2D(conv3_3, 256, kernel_size=3, stride=1, name='conv3_4')
    # pool3 = MaxPool2D(conv3_4, kernel_size=2, stride=2, name='max_pool3')

    conv4_1 = Conv2D(conv3_4, 512, kernel_size=3, stride=1, name='conv3_1')
    conv4_2 = Conv2D(conv4_1, 512, kernel_size=3, stride=1, name='conv3_2')
    conv4_3 = Conv2D(conv4_2, 512, kernel_size=3, stride=1, name='conv3_3')
    conv4_4 = Conv2D(conv4_3, 512, kernel_size=3, stride=1, name='conv3_4')
    # pool4 = MaxPool2D(conv4_4, kernel_size=2, stride=2, name='max_pool3')

    conv5_1 = Conv2D(conv4_4, 512, kernel_size=3, stride=1, name='conv3_1')
    conv5_2 = Conv2D(conv5_1, 512, kernel_size=3, stride=1, name='conv3_2')
    conv5_3 = Conv2D(conv5_2, 512, kernel_size=3, stride=1, name='conv3_3')
    conv5_4 = Conv2D(conv5_3, 512, kernel_size=3, stride=1, name='conv3_4')
    pool5 = MaxPool2D(conv5_4, kernel_size=2, stride=2, name='max_pool3')

    flatten = tf.reshape(pool5, [-1, 4 * 4 * 512])
    fc6 = FullyConnected(flatten, 4096, name='fc6')
    dropout1 = tf.nn.dropout(fc6, rate=1 - keep_prob)

    fc7 = FullyConnected(dropout1, 4096, name='fc7')
    dropout2 = tf.nn.dropout(fc7, rate=1 - keep_prob)

    fc8 = FullyConnected(dropout2, classes, name='fc8')

    return fc8
