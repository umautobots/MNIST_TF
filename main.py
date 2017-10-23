import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from util import build_model
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

mnist = input_data.read_data_sets('./data', one_hot=True)

x = tf.placeholder(tf.float32, [None, 28**2], name='images')
y = tf.placeholder(tf.float32, [None, 10], name='labels')

with tf.variable_scope('model'):
    accu_train, loss_train = build_model(x, y)

with tf.variable_scope('model', reuse=True):
    accu_test, loss_test = build_model(x, y, training=False)

with tf.variable_scope('optimizer'):
    # Decreasing learning rate
    lr = tf.Variable(2e-3, trainable=False, name='learning_rate')
    lr_update = tf.assign(lr, 0.5 * lr)

    # Update the moving average/variance in batch normalization layers
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_train)

# Summaries
sum_lr = tf.summary.scalar('misc/learning_rate', lr)

sum_accu_train = tf.summary.scalar('train/accuracy', accu_train)
sum_loss_train = tf.summary.scalar('train/loss', loss_train)
sum_train = tf.summary.merge([sum_accu_train, sum_loss_train])

sum_accu_test = tf.summary.scalar('test/accuracy', accu_test)
sum_loss_test = tf.summary.scalar('test/loss', loss_test)
sum_test = tf.summary.merge([sum_accu_test, sum_loss_test])

sum_weights = tf.summary.merge([tf.summary.histogram(var.name, var) for var in tf.trainable_variables()])

batch_size = 500
num_steps = 10000

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    log_dir = './logs/{:d}'.format(int(time()))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    t0 = time()
    for step in range(num_steps):
        if step % (num_steps // 100) == 0 or step == num_steps - 1:
            a_test, s_test, s_weights = sess.run([accu_test, sum_test, sum_weights],
                                                 feed_dict={x: mnist.test.images, y: mnist.test.labels})
            t = time() - t0
            m, s = divmod(t, 60)
            h, m = divmod(m, 60)
            print('[{:6d}/{:6d}] Time: [{:02d}:{:02d}:{:02d}], Test accuracy: {:.4f}'.format(
                  step, num_steps, int(h), int(m), int(s), a_test))
            writer.add_summary(s_test, step)
            writer.add_summary(s_weights, step)

        img, lbl = mnist.train.next_batch(batch_size)
        _, s_train, s_lr = sess.run([optim, sum_train, sum_lr],
                                    feed_dict={x: img, y: lbl})
        writer.add_summary(s_train, step)
        writer.add_summary(s_lr, step)

        if (step + 1) % (num_steps // 4) == 0:
            sess.run(lr_update)
