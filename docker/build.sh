#!/bin/bash

docker build \
    --network host \
    -t mnist_tf:2.3.1 \
    ./docker
