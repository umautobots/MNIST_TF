# Using convolutional neural network to classify handwritten digits (MNIST) in TensorFlow.

# Run on regular environments (e.g. Ubuntu 16.04)

## Dependencies
- Python >= 3.6
- TensorFlow >= 1.13.1 with GPU support

## To start training, run
```
$ python3 main.py
```

## To visualize the training/test loss, run
```
$ tensorboard --logdir=logs
```
and TensorBoard will be availabel at `0.0.0.0:6006`

# Run with [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) (recommended)

## Dependencies
- NVIDIA Docker

## To start training, run
```
$ ./train.sh
```

## To visualize the training/test loss, run
```
$ ./tensorboard.sh
```
and TensorBoard will be availabel at `0.0.0.0:6006`
