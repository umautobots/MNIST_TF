#! /usr/bin/env python3
import os
from datetime import datetime
import argparse
import numpy as np
import tensorflow as tf
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=50,
                    help='Number of epochs. (default: 50)')
parser.add_argument('--batch_size', type=int, default=500,
                    help='Batch size in each training step. (default: 500)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate. (default: 1e-3)')
parser.add_argument('--train_ratio', type=float, default=0.75,
                    help='Fraction of data for training. (default: 0.75)')
parser.add_argument('--tag', type=str, default='',
                    help='Tag')
parser.add_argument('--fft', dest='fft', action='store_true')
parser.add_argument('--fashion', dest='fashion', action='store_true')
args = parser.parse_args()

if args.fashion:
    mnist = tf.keras.datasets.fashion_mnist
else:
    mnist = tf.keras.datasets.mnist

(x, y), _ = mnist.load_data()

idx_train = np.arange(x.shape[0]) < args.train_ratio * x.shape[0]

assert 0 < np.mean(idx_train) < 1

np.random.shuffle(idx_train)
x_train, y_train = x[idx_train], y[idx_train]
x_val, y_val = x[~idx_train], y[~idx_train]

utils.allow_gpu_memory_growth()

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = utils.build_model(fft=args.fft)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

model.summary()

for layer in model.layers:
    if 'conv2d' in layer.name:
        weights = layer.get_weights()
        for w in weights:
            print(f'Name: {layer.name}, shape: {w.shape}')

if len(args.tag):
    log_dir = f'./logs/{args.tag:s}'
else:
    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log_dir = f'./logs/{now:s}'

# TensorBoard
tb_callback = tf.keras.callbacks.TensorBoard(
   log_dir=log_dir,
   histogram_freq=1
)

# Checkpoint
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(log_dir, 'cp.ckpt'),
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
