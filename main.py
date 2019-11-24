#! /usr/bin/python3
from datetime import datetime
import argparse
import tensorflow as tf
from tensorflow import keras
import util


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs',
                    help='Number of epochs. (default: 25)',
                    type=int, default=25)
parser.add_argument('--batch_size',
                    help='Batch size in each training step. (default: 1024)',
                    type=int, default=1024)
parser.add_argument('--lr',
                    help='Learning rate. (default: 1e-3)',
                    type=float, default=1e-3)
parser.add_argument('--fashion', dest='fashion', action='store_true')
parser.set_defaults(fashion=False)
args = parser.parse_args()

# allow memory to grow when needed
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f'{len(gpus):d} Physical GPU(s), {len(logical_gpus):d} Logical GPU(s)')
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

if args.fashion:
    mnist = keras.datasets.fashion_mnist
else:
    mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = 0.5 * (x_train - 127.5)
x_test = 0.5 * (x_test - 127.5)

model = util.build_model()

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(lr=args.lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
tb_callback = keras.callbacks.TensorBoard(
   log_dir=f'./logs/{now:s}',
   histogram_freq=1
)

model.fit(
    x_train,
    y_train,
    epochs=args.num_epochs,
    batch_size=args.batch_size,
    validation_data=(x_test, y_test),
    callbacks=[tb_callback]
)
