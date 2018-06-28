#! /bin/bash
nvidia-docker run -it --rm \
  -v $PWD:/root/MNIST_TF \
  -v /mnt:/mnt:ro \
  -w /root/MNIST_TF \
  tensorflow/tensorflow:latest-devel-gpu-py3 \
  python3 main.py --gpu 0
