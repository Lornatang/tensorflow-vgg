import tensorflow as tf
import numpy as np


# print layer information
def print_layer(t):
    print(f"{t.op.name} {t.get_shape().as_list()} \n")


# conv layer op
def conv2d(x, out, name, kernel_size, stride):
    """

    :param x: input tensor.
    :param out: output tensor.
    :param name: layer name.
    :param kernel_size: kernel size.
    :param stride: stride size.
    :return: activation.
    """
    input_x = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[kernel_size, kernel_size, input_x, out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(x, kernel, (1, stride, stride, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        print_layer(activation)
        return activation


# define fully connected
def fully_connected(x, out, name):
    """

    :param x: input tensor.
    :param out: output tensor.
    :param name: layer name.
    :return: activation
    """
    input_x = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[input_x, out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.Variable(tf.constant(0.1, shape=[out], dtype=tf.float32, name='b'))
        activation = tf.nn.relu_layer(input_x, kernel, biases, name=scope)
        print_layer(activation)
        return activation


# define maxpool layer
def mpool_op(input_op,name,kh,kw,dh,dw):
    return tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding='SAME',name=name)
