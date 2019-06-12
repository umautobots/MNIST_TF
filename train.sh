#! /bin/bash

mkdir -p $HOME/.keras

docker run -it --rm \
  --runtime=nvidia \
  --name=mnist \
  -u $(id -u):$(id -g) \
  -e HOME=$HOME \
  -v $PWD:$HOME/mnist_tf \
  -v $HOME/.keras:$HOME/.keras \
  -w $HOME/mnist_tf \
  tensorflow/tensorflow:1.13.1-gpu-py3 \
  bash -c "python3 main.py && rm -rf __pycache__"
