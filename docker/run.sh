#! /bin/bash

if [ -z $1 ] ; then
    GPU=all
else
    GPU=$1
fi

HOME_DIR=/home/$(whoami)

docker run -it --rm \
    --gpus '"device='$GPU'"' \
    --name 'mnist-tf' \
    -u $(id -u):$(id -g) \
    -e HOME=$HOME_DIR \
    -v $(pwd)/data:$HOME_DIR/.keras \
    -v $(pwd):$HOME_DIR/mnist_tf \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -w $HOME_DIR/mnist_tf \
    tensorflow/tensorflow:2.2.0-gpu
