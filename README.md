# Using convolutional neural network to classify handwritten digits (MNIST) in TensorFlow.

# Run on regular environments (e.g. Ubuntu 18.04)

## Dependencies
- Python >= 3.6
- TensorFlow >= 2.0.0 with GPU support

## To start training, run
```
$ export CUDA_VISIBLE_DEVICES=0
$ python3 main.py
```

## To visualize the training/test loss, run
```
$ tensorboard --logdir=logs
```
and TensorBoard will be availabel at `localhost:6006`

# Run with [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) (recommended)

## Dependencies
- NVIDIA Docker

## To start training, run
```
$ ./train.sh    # train on GPU0 (default)
$ ./train.sh 2  # train on GPU2
```

## To visualize the training/test loss, run
```
$ ./tensorboard.sh
```
and TensorBoard will be availabel at `localhost:6006`
