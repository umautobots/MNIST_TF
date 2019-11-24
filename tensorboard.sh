#! /bin/bash
if [ -z $1 ] ; then
  PORT=6006
else
  PORT=$1
fi

docker run -it --rm \
  -v $PWD/logs:/logs:ro \
  -p $PORT:$PORT \
  tensorflow/tensorflow:2.0.0-gpu-py3 \
  tensorboard --logdir=/logs --port=$PORT
