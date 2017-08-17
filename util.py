import tensorflow as tf

def build_model(image, label, training=True):
    with tf.variable_scope('reshape'):
        x = tf.reshape(image, [-1, 28, 28, 1])
        x = 2 * x - 1
        y = tf.reshape(label, [-1, 1, 1, 10])

    for k in range(3):
        with tf.variable_scope('layer{:d}'.format(k + 1)):
            x = tf.layers.conv2d(x, 2**k * 64, 3, 2, use_bias=False, padding='same')
            x = tf.layers.batch_normalization(x, scale=False, training=training)
            x = tf.nn.relu(x)
            x = tf.layers.dropout(x, training=training)

    # This is equivalent to a fully connnected layer
    x = tf.layers.conv2d(x, 10, x.shape[1:3], name='logits')

    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=x))

    with tf.variable_scope('benchmark'):
        predictions = tf.argmax(x, axis=-1, name='predictions')
        correct_predictions = tf.equal(predictions, tf.argmax(y, -1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

    return accuracy, loss
