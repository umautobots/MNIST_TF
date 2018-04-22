import os
import numpy as np
import tensorflow as tf


def build_model(x, y, training=True):
    for k in range(3):
        with tf.variable_scope('layer{:d}'.format(k + 1)):
            x = tf.layers.conv2d(x, 2**k * 64, 3,
                                 strides=2, use_bias=False, padding='same')
            x = tf.layers.batch_normalization(x,
                                              scale=False, training=training)
            x = tf.nn.relu(x)
            x = tf.layers.dropout(x, training=training)

    with tf.variable_scope('logits'):
        numel = np.prod(x.shape.as_list()[1:])
        x = tf.reshape(x, [-1, numel])
        x = tf.layers.dense(x, 10)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=x,
                                                  scope='loss')

    with tf.variable_scope('benchmark'):
        predictions = tf.argmax(x, axis=-1, name='predictions')
        correct_predictions = tf.equal(predictions, y)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32),
                                  name='accuracy')

    return accuracy, loss


def preprocess(image, label):
    x = tf.reshape(image, [28, 28, 1])
    x = tf.cast(x, tf.float32) / 127.5 - 1

    y = tf.squeeze(label)
    y = tf.cast(y, tf.int64)

    return x, y


def load_dataset(data_dir, dataset):
    if dataset.lower() == 'train':
        fn_img = 'train-images-idx3-ubyte'
        fn_lbl = 'train-labels-idx1-ubyte'
    elif dataset.lower() == 'test':
        fn_img = 't10k-images-idx3-ubyte'
        fn_lbl = 't10k-labels-idx1-ubyte'
    else:
        raise ValueError('Unrecognized set `{}`'.format(dataset))

    raw_img = np.fromfile(os.path.join(data_dir, fn_img), dtype=np.uint8)
    num_img = np.sum(raw_img[4:8] * 256**np.arange(3, -1, -1))

    raw_lbl = np.fromfile(os.path.join(data_dir, fn_lbl), dtype=np.uint8)
    num_lbl = np.sum(raw_lbl[4:8] * 256**np.arange(3, -1, -1))

    assert num_img == num_lbl

    height = np.sum(raw_img[8:12] * 256**np.arange(3, -1, -1))
    width = np.sum(raw_img[12:16] * 256**np.arange(3, -1, -1))

    img = raw_img[16:].reshape([num_img, height, width])
    lbl = raw_lbl[8:]

    return num_img, tf.data.Dataset.from_tensor_slices((img, lbl))
