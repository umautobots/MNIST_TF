import os
from time import time
import argparse
import tensorflow as tf
import util


parser = argparse.ArgumentParser()

parser.add_argument('--num_steps',
                    help='Number of training steps. (default: 10000)',
                    type=int, default=10000)
parser.add_argument('--batch_size',
                    help='Batch size in each training step. (default: 256)',
                    type=int, default=256)
parser.add_argument('--gpu',
                    help='Which GPU to use. (default: 0)',
                    type=str, default='0')
parser.add_argument('--lr',
                    help='Learning rate. (default: 2e-3)',
                    type=float, default=2e-3)
parser.add_argument('--print_step',
                    help='Number of steps for printing info. (default: 100)',
                    type=int, default=100)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

with tf.variable_scope('input/train'):
    _, ds = util.load_dataset('data', 'train')
    ds = ds.shuffle(10000).repeat()
    ds = ds.map(util.preprocess, num_parallel_calls=64).batch(args.batch_size)
    ds = ds.prefetch(10)
    x_train, y_train = ds.make_one_shot_iterator().get_next()
    print(x_train.shape, y_train.shape)

with tf.variable_scope('input/test'):
    num_test, ds = util.load_dataset('data', 'test')
    ds = ds.repeat()
    ds = ds.map(util.preprocess, num_parallel_calls=64).batch(num_test)
    ds = ds.prefetch(10)
    x_test, y_test = ds.make_one_shot_iterator().get_next()

with tf.variable_scope('model'):
    accu_train, loss_train = util.build_model(x_train, y_train)

with tf.variable_scope('model', reuse=True):
    accu_test, loss_test = util.build_model(x_test, y_test, training=False)

with tf.variable_scope('optimizer'):
    # Decreasing learning rate
    lr = tf.Variable(args.lr, trainable=False, name='learning_rate')
    lr_update = tf.assign(lr, 0.5 * lr)

    # Update the moving average/variance in batch normalization layers
    # https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(lr).minimize(loss_train)

# Summaries
sum_lr = tf.summary.scalar('misc/learning_rate', lr)

sum_accu_train = tf.summary.scalar('train/accuracy', accu_train)
sum_loss_train = tf.summary.scalar('train/loss', loss_train)
sum_train = tf.summary.merge([sum_accu_train, sum_loss_train])

sum_accu_test = tf.summary.scalar('test/accuracy', accu_test)
sum_loss_test = tf.summary.scalar('test/loss', loss_test)
sum_test = tf.summary.merge([sum_accu_test, sum_loss_test])

vars = tf.trainable_variables()
sum_weights = tf.summary.merge([tf.summary.histogram(v.name, v) for v in vars])

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    log_dir = './logs/{:d}'.format(int(time()))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    t0 = time()
    for step in range(args.num_steps):
        if step % args.print_step == 0 or step == args.num_steps - 1:
            a_test, s_test, s_weights = sess.run([accu_test, sum_test, sum_weights])
            t = time() - t0
            m, s = divmod(t, 60)
            h, m = divmod(m, 60)
            print('[{:6d}/{:6d}] Time: [{:02d}:{:02d}:{:02d}], Test accuracy: {:.4f}'.format(
                  step, args.num_steps, int(h), int(m), int(s), a_test))
            writer.add_summary(s_test, step)
            writer.add_summary(s_weights, step)

        _, s_train, s_lr = sess.run([train_op, sum_train, sum_lr])
        writer.add_summary(s_train, step)
        writer.add_summary(s_lr, step)

        if (step + 1) % (args.num_steps // 4) == 0:
            sess.run(lr_update)
