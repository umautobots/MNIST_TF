#! /bin/bash

if [ -z $1 ] ; then
    GPU=all
else
    GPU=$1
fi

docker run -it --rm \
    --gpus '"device='$GPU'"' \
    --name 'mnist-tf' \
    --hostname $(hostname) \
    -u $(id -u):$(id -g) \
    -e HOME \
    -v $HOME/.bash_history:$HOME/.bash_history \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -v $(pwd)/data:$HOME/.keras \
    -v $(pwd):/mnist_tf \
    -w /mnist_tf \
    tensorflow/tensorflow:2.3.0-gpu
