#! /bin/bash
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P $PWD/data
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P $PWD/data
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P $PWD/data
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P $PWD/data

gunzip -d $PWD/data/*.gz
