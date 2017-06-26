nvidia-docker run -it --rm \
  -v `pwd`:/root/MNIST_TF \
  -w /root/MNIST_TF \
  tensorflow/tensorflow:1.2.0-devel-gpu-py3 \
  python3 main.py
