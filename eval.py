#! /usr/bin/env python3
import os
from glob import glob
import argparse
from tensorflow import keras
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--idx_ckpt', type=int, default=-1,
                    help='Index of the checkpoint to load. (default: -1)')
parser.add_argument('--fft', dest='fft', action='store_true')
parser.add_argument('--fashion', dest='fashion', action='store_true')
args = parser.parse_args()

checkpoints = sorted(glob('logs/*'))
if len(checkpoints):
    [print(f'{idx}: {ckpt}') for idx, ckpt in enumerate(checkpoints)]
else:
    raise ValueError('Can not find any checkpoint.')

if args.fashion:
    mnist = keras.datasets.fashion_mnist
else:
    mnist = keras.datasets.mnist

_, (x_test, y_test) = mnist.load_data()

utils.allow_gpu_memory_growth()

model = utils.build_model(fft=args.fft)
model.compile(
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

loss, acc = model.evaluate(x_test, y_test, verbose=0)

print(f'[!] Untrained model: accuracy = {100 * acc:5.2f}%')

print(f'Loading {checkpoints[args.idx_ckpt]}')
model.load_weights(os.path.join(checkpoints[args.idx_ckpt], 'cp.ckpt'))

loss, acc = model.evaluate(x_test, y_test, verbose=0)

print(f'[!] Restored model: accuracy = {100 * acc:5.2f}%')
