# Using convolutional neural network to classify handwritten digits (MNIST) in TensorFlow.

Here we have code for
- visualizing loss, accuracy, and the graph using TensorBoard
- constructing a network with high level functions in `tf.layers`
- implementing decreasing learning rate

This implementation achieves an accuracy of ~99.4% after 10k steps.

# Run on regular environments (e.g. Ubuntu 16.04)

## Dependencies
- Python >= 3.5.2
- CUDA 8
- CUDNN 5.1
- TensorFlow >= 1.0.0

## To start training, run
```
$ python3 main.py
```

## To visualize the training/test loss, run
```
$ tensorboard --logdir=./logs
```

# Run with [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) (recommended)

## Dependencies
- NVIDIA Docker

## To start training, run
```
$ ./docker_train.sh
```

## To visualize the training/test loss, run
```
$ ./docker_tensorboard.sh 6006
```
