#! /usr/bin/python3
import os
from datetime import datetime
import argparse
import tensorflow as tf
from tensorflow.keras import datasets, optimizers, backend
from tensorflow.keras.callbacks import TensorBoard
import util


gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
backend.set_session(sess)


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs',
                    help='Number of epochs. (default: 25)',
                    type=int, default=25)
parser.add_argument('--batch_size',
                    help='Batch size in each training step. (default: 1024)',
                    type=int, default=1024)
parser.add_argument('--gpu',
                    help='Which GPU to use. (default: 0)',
                    type=str, default='0')
parser.add_argument('--lr',
                    help='Learning rate. (default: 1e-3)',
                    type=float, default=1e-3)
parser.add_argument('--fashion', dest='fashion', action='store_true')
parser.set_defaults(fashion=False)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.fashion:
    mnist = datasets.fashion_mnist
else:
    mnist = datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = 0.5 * (x_train - 127.5)
x_test = 0.5 * (x_test - 127.5)

model = util.build_model()

model.compile(
    optimizer=optimizers.Adam(lr=args.lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
tb = TensorBoard(
    log_dir='./logs/{:s}'.format(now),
    update_freq='batch',
    histogram_freq=1
)

model.fit(
    x_train, y_train,
    epochs=args.num_epochs,
    batch_size=args.batch_size,
    validation_data=(x_test, y_test),
    callbacks=[tb]
)

sess.close()
