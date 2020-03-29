#! /bin/bash

if [ -z $1 ] ; then
  GPU=all
else
  GPU=$1
fi

mkdir -p $HOME/.keras
WORKDIR=/mnist_tf

docker run -it --rm \
  --gpus '"device='$GPU'"' \
  --name 'mnist-gpu-'$GPU \
  -u $(id -u):$(id -g) \
  -e HOME=$HOME \
  -v $HOME/.keras:$HOME/.keras \
  -v $(pwd):$WORKDIR \
  -v /etc/timezone:/etc/timezone:ro \
  -v /etc/localtime:/etc/localtime:ro \
  -w $WORKDIR \
  tensorflow/tensorflow:2.1.0-gpu-py3 \
  python3 train.py
