nvidia-docker run -it --rm \
  -v `pwd`/logs:/logs:ro \
  -p $1:6006 \
  tensorflow/tensorflow:1.2.0-devel-gpu-py3 \
  tensorboard --logdir=/logs
