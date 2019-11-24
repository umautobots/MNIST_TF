#! /bin/bash

if [ -z $1 ] ; then
  GPU=0
else
  GPU=$1
fi

mkdir -p $HOME/.keras

docker run -it --rm \
  --gpus='"device='$GPU'"' \
  --name='mnist-gpu'$GPU \
  -u $(id -u):$(id -g) \
  -e HOME=$HOME \
  -v $PWD:$HOME/mnist_tf \
  -v $HOME/.keras:$HOME/.keras \
  -w $HOME/mnist_tf \
  tensorflow/tensorflow:2.0.0-gpu-py3 \
  bash -c "python3 main.py && rm -rf __pycache__"
