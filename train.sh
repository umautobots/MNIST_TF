#! /bin/bash

if [ -z $1 ] ; then
  GPU=0
else
  GPU=$1
fi

mkdir -p $HOME/.keras
WORKDIR=/mnist_tf

docker run -it --rm \
  --gpus='"device='$GPU'"' \
  --name='mnist-gpu'$GPU \
  -u $(id -u):$(id -g) \
  -e HOME=$HOME \
  -v $HOME/.keras:$HOME/.keras \
  -v $PWD/main.py:$WORKDIR/main.py:ro \
  -v $PWD/util.py:$WORKDIR/util.py:ro \
  -v $PWD/logs:$WORKDIR/logs \
  -v $PWD/checkpoints:$WORKDIR/checkpoints \
  -v /etc/timezone:/etc/timezone:ro \
  -v /etc/localtime:/etc/localtime:ro \
  -w $WORKDIR \
  tensorflow/tensorflow:2.0.0-gpu-py3 \
  python3 train.py
