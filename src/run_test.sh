#!/bin/bash

set -x

python -u test.py +experiment=test cfg_dir=/home/aghinassi/Desktop/checkpoints/basic_gan_lessadv window_size=128 && killall python