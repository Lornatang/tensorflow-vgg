from datetime import datetime
from vgg19 import *

batch_size = 64
lr = 1e-4
classes = 10
max_steps = 50000


def read_and_decode(filename):
    """

    :param filename: tf records file name.
    :return: image and labels.
    """
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'data': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['data'], tf.uint8)
    img = tf.reshape(img, [32, 32, 3])
    # trans float32 and norm
    img = tf.cast(img, tf.float32)  # * (1. / 255)
    label = tf.cast(features['label'], tf.int64)
    return img, label


def train():
    X = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='input')
    y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name='label')
    keep_prob = tf.placeholder(tf.float32)
    output = vgg19(X, keep_prob, classes)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(y, 1)), tf.float32))

    images, labels = read_and_decode('train.tfrecords')
    img_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                    batch_size=batch_size,
                                                    capacity=200,
                                                    min_after_dequeue=100)
    label_batch = tf.one_hot(label_batch, classes, 1, 0)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(max_steps):
            batch_x, batch_y = sess.run([img_batch, label_batch])
            _, loss_val = sess.run([train_step, loss], feed_dict={X: batch_x, y: batch_y, keep_prob: 0.8})
            if i % 10 == 0:
                train_arr = accuracy.eval(feed_dict={X: batch_x, y: batch_y, keep_prob: 1.0})
                print(f"{datetime.now()}: Step [%d/{max_steps}]  Loss : {i:.8f}, training accuracy :  {train_arr:.4g}")
            if (i + 1) == max_steps:
                saver.save(sess, './model/model.ckpt', global_step=i)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()
