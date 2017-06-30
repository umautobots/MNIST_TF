nvidia-docker run -it --rm \
  -v `pwd`/logs:/logs:ro \
  -p $1:$1 \
  tensorflow/tensorflow:1.2.1-devel-gpu-py3 \
  tensorboard --logdir=/logs --port=$1
