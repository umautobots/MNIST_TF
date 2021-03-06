#! /bin/bash
if [ -z $1 ] ; then
  PORT=6006
else
  PORT=$1
fi

docker run -it --rm \
  --hostname $(hostname) \
  -v $(pwd)/logs:/logs:ro \
  -p $PORT:$PORT \
  -v /etc/timezone:/etc/timezone:ro \
  -v /etc/localtime:/etc/localtime:ro \
  mnist_tf:2.3.1 \
  tensorboard --logdir=/logs --port=$PORT --bind_all
