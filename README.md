# Using convolutional neural network to classify handwritten digits (MNIST) in TensorFlow.

Here we have code for
- visualizing loss, accuracy, and the graph using TensorBoard
- loading the dataset using `tf.data` API
- constructing a network with high level APIs in `tf.layers`
- implementing decreasing learning rate

This implementation achieves an accuracy of ~99.4% after 10k steps.

# Run on regular environments (e.g. Ubuntu 16.04)

## Dependencies
- Python >= 3.5.2
- TensorFlow >= 1.0.0 with GPU support

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
