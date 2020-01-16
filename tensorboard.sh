#! /bin/bash
if [ -z $1 ] ; then
  PORT=6006
else
  PORT=$1
fi

docker run -it --rm \
  -v $PWD/logs:/logs:ro \
  -p $PORT:$PORT \
  -v /etc/timezone:/etc/timezone:ro \
  -v /etc/localtime:/etc/localtime:ro \
  tensorflow/tensorflow:2.1.0-gpu-py3 \
  tensorboard --logdir=/logs --port=$PORT --bind_all
