import tensorflow as tf
import numpy as np
import time
import os
import sys
import pickle
import random


class_num = 10
image_size = 32
img_channels = 3
iterations = 200
batch_size = 250
total_epoch = 164
weight_decay = 0.0003
dropout_rate = 0.5
momentum_rate = 0.9
log_save_path = './vgg_logs'
model_save_path = './model/'


def download_data():
    dirname = 'cifar10-dataset'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    fname = './CAFIR-10_data/cifar-10-python.tar.gz'
    fpath = './' + dirname

    if os.path.exists(fpath) or os.path.isfile(fname):
        download = False
        print("DataSet already exist!")
    else:
        download = True
    if download:
        print('Downloading data from', origin)
        import urllib.request
        import tarfile

        def reporthook(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            percent = min(int(count * block_size * 100 / total_size), 100)
            sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                             (percent, progress_size / (1024 * 1024), speed, duration))
            sys.stdout.flush()

        urllib.request.urlretrieve(origin, fname, reporthook)
        print('Download finished. Start extract!', origin)
        if fname.endswith("tar.gz"):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall()
            tar.close()
        elif fname.endswith("tar"):
            tar = tarfile.open(fname, "r:")
            tar.extractall()
            tar.close()


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels


def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    data = data.reshape([-1, img_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels


def prepare_data():
    print("======Loading data======")
    download_data()
    data_dir = './cifar10-dataset'
    image_dim = image_size * image_size * img_channels
    meta = unpickle(data_dir + '/batches.meta')

    label_names = meta[b'label_names']
    label_count = len(label_names)
    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_data, train_labels = load_data(train_files, data_dir, label_count)
    test_data, test_labels = load_data(['test_batch'], data_dir, label_count)

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    print("======Prepare Finished======")

    return train_data, train_labels, test_data, test_labels


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(input, k_size=1, stride=1, name=None):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                          padding='SAME', name=name)


def batch_norm(input):
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        is_training=train_flag, updates_collections=None)


def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                                    nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def data_preprocessing(x_train, x_test):

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, x_test


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32, 32], 4)
    return batch


def learning_rate_schedule(epoch_num):
    if epoch_num < 81:
        return 0.1
    elif epoch_num < 121:
        return 0.01
    else:
        return 0.001


def run_testing(sess, ep):
    acc = 0.0
    loss = 0.0
    pre_index = 0
    add = 1000
    for it in range(10):
        batch_x = test_x[pre_index:pre_index + add]
        batch_y = test_y[pre_index:pre_index + add]
        pre_index = pre_index + add
        loss_, acc_ = sess.run([cross_entropy, accuracy],
                               feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: False})
        loss += loss_ / 10.0
        acc += acc_ / 10.0
    summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=loss),
                                tf.Summary.Value(tag="test_accuracy", simple_value=acc)])
    return acc, loss, summary


if __name__ == '__main__':

    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = data_preprocessing(train_x, test_x)

    # define placeholder x, y_ , keep_prob, learning_rate
    x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, class_num])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    # build_network

    W_conv1_1 = tf.get_variable('conv1_1', shape=[3, 3, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_1 = bias_variable([64])
    output = tf.nn.relu(batch_norm(conv2d(x, W_conv1_1) + b_conv1_1))

    W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_2 = bias_variable([64])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv1_2) + b_conv1_2))
    output = max_pool(output, 2, 2, "pool1")

    W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, 64, 128], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_1 = bias_variable([128])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_1) + b_conv2_1))

    W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, 128, 128], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_2 = bias_variable([128])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_2) + b_conv2_2))
    output = max_pool(output, 2, 2, "pool2")

    W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 128, 256], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_1 = bias_variable([256])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_1) + b_conv3_1))

    W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_2 = bias_variable([256])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_2) + b_conv3_2))

    W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_3 = bias_variable([256])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_3) + b_conv3_3))

    W_conv3_4 = tf.get_variable('conv3_4', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_4 = bias_variable([256])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_4) + b_conv3_4))
    output = max_pool(output, 2, 2, "pool3")

    W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 256, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_1 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_1) + b_conv4_1))

    W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_2 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_2) + b_conv4_2))

    W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_3 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_3) + b_conv4_3))

    W_conv4_4 = tf.get_variable('conv4_4', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_4 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_4)) + b_conv4_4)
    output = max_pool(output, 2, 2)

    W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_1 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_1) + b_conv5_1))

    W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_2 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_2) + b_conv5_2))

    W_conv5_3 = tf.get_variable('conv5_3', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_3 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_3) + b_conv5_3))

    W_conv5_4 = tf.get_variable('conv5_4', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_4 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_4) + b_conv5_4))

    # output = tf.contrib.layers.flatten(output)
    output = tf.reshape(output, [-1, 2 * 2 * 512])

    W_fc1 = tf.get_variable('fc1', shape=[2048, 4096], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc1 = bias_variable([4096])
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc1) + b_fc1))
    output = tf.nn.dropout(output, keep_prob)

    W_fc2 = tf.get_variable('fc7', shape=[4096, 4096], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc2 = bias_variable([4096])
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc2) + b_fc2))
    output = tf.nn.dropout(output, keep_prob)

    W_fc3 = tf.get_variable('fc3', shape=[4096, 10], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc3 = bias_variable([10])
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc3) + b_fc3))
    # output  = tf.reshape(output,[-1,10])

    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True).\
        minimize(cross_entropy + l2 * weight_decay)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # initial an saver to save model
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_save_path, sess.graph)

        # epoch = 164
        # make sure [bath_size * iteration = data_set_number]

        for ep in range(1, total_epoch + 1):
            lr = learning_rate_schedule(ep)
            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0
            start_time = time.time()

            print("\n epoch %d/%d:" % (ep, total_epoch))

            for it in range(1, iterations + 1):
                batch_x = train_x[pre_index:pre_index + batch_size]
                batch_y = train_y[pre_index:pre_index + batch_size]

                batch_x = data_augmentation(batch_x)

                _, batch_loss = sess.run([train_step, cross_entropy],
                                         feed_dict={x: batch_x, y_: batch_y, keep_prob: dropout_rate,
                                                    learning_rate: lr, train_flag: True})
                batch_acc = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: True})

                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += batch_size

                if it == iterations:
                    train_loss /= iterations
                    train_acc /= iterations

                    loss_, acc_ = sess.run([cross_entropy, accuracy],
                                           feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: True})
                    train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss),
                                                      tf.Summary.Value(tag="train_accuracy", simple_value=train_acc)])

                    val_acc, val_loss, test_summary = run_testing(sess, ep)

                    summary_writer.add_summary(train_summary, ep)
                    summary_writer.add_summary(test_summary, ep)
                    summary_writer.flush()

                    print("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, "
                          "train_acc: %.4f, test_loss: %.4f, test_acc: %.4f"
                          % (it, iterations, int(time.time() - start_time), train_loss, train_acc, val_loss, val_acc))
                else:
                    print("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f"
                          % (it, iterations, train_loss / it, train_acc / it), end='\r')

        save_path = saver.save(sess, model_save_path)
        print("Model saved in file: %s" % save_path)
