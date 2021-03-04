#!/bin/bash

sudo docker build -t fastai-v3 . && sudo docker run --rm -it -v /mnt/btrfs-data/venvs/ml-tutorials/data/chihuahua-or-muffin-fastai-v3-docker-container-uploaded-images:/tmp/saved-images -p 55561:55561 fastai-v3
