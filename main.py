#! /usr/bin/python3
from datetime import datetime
import argparse
import numpy as np
from tensorflow import keras
import util


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=50,
                    help='Number of epochs. (default: 50)')
parser.add_argument('--batch_size', type=int, default=500,
                    help='Batch size in each training step. (default: 500)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate. (default: 1e-3)')
parser.add_argument('--train_ratio', type=float, default=0.75,
                    help='Fraction of data for training. (default: 0.75)')
parser.add_argument('--fashion', dest='fashion', action='store_true')
args = parser.parse_args()

if args.fashion:
    mnist = keras.datasets.fashion_mnist
else:
    mnist = keras.datasets.mnist

(x, y), _ = mnist.load_data()

idx_train = np.arange(x.shape[0]) < args.train_ratio * x.shape[0]

assert 0 < np.mean(idx_train) < 1

np.random.shuffle(idx_train)
x_train, y_train = x[idx_train], y[idx_train]
x_val, y_val = x[~idx_train], y[~idx_train]

util.allow_gpu_memory_growth()

model = util.build_model()
model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(lr=args.lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# TensorBoard
now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
tb_callback = keras.callbacks.TensorBoard(
   log_dir=f'./logs/{now:s}',
   histogram_freq=1
)

# Checkpoint
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=f'./checkpoints/{now:s}/cp.ckpt',
    save_weights_only=True,
    verbose=1
)

model.fit(
    x_train,
    y_train,
    epochs=args.num_epochs,
    batch_size=args.batch_size,
    validation_data=(x_val, y_val),
    callbacks=[tb_callback, cp_callback]
)
