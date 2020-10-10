#! /bin/bash

if [ -z $1 ] ; then
    GPU=all
else
    GPU=$1
fi

mkdir -p .fake_home

docker run -it --rm \
    --gpus '"device='$GPU'"' \
    --name 'mnist_tf' \
    --hostname $(hostname) \
    -u $(id -u):$(id -g) \
    -e HOME \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -v $(pwd)/.fake_home:$HOME \
    -v $(pwd):/mnist_tf \
    -w /mnist_tf \
    mnist_tf:2.3.1
