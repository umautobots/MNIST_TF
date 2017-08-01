if [ -z $1 ] ; then
  PORT=6006
else
  PORT=$1
fi

nvidia-docker run -it --rm \
  -v `pwd`/logs:/logs:ro \
  -p $PORT:$PORT \
  tensorflow/tensorflow:latest-devel-gpu-py3 \
  tensorboard --logdir=/logs --port=$PORT
