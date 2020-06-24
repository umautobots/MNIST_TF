# Using a convolutional neural network to classify handwritten digits (MNIST) in [TensorFlow](https://www.tensorflow.org/).

# Run on regular environments (e.g. Ubuntu 18.04)

## Dependencies
- NVIDIA driver >= 418
- Python >= 3.6
- TensorFlow >= 2.2.0 with GPU support (CUDA 10.1 & cuDNN 7)

After installing [CUDA 10.1](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [cuDNN 7](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html), run the following commands to install TensorFlow:

```
$ wget https://bootstrap.pypa.io/get-pip.py
$ python3 get-pip.py --user --upgrade
$ rm -rf get-pip.py
$ pip3 install --user tensorflow
```

## Training

```
$ ./train.py
```

## To visualize the training/test loss, run

```
$ tensorboard --logdir=logs
```

and TensorBoard will be available at `localhost:6006`

# Run with [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) (recommended)

## Dependencies
- NVIDIA driver >= 418
- NVIDIA Docker

## Launch a container

```
$ ./docker/run.sh
```

## Training

```
$ ./train.py        # run this in the container
```

## To visualize the training/test loss, run

```
$ ./docker/tensorboard.sh
```

and TensorBoard will be available at `localhost:6006`
