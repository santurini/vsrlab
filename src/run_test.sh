#!/bin/bash

set -x

python -u test_torch.py +experiment=test
python -u test_torch.py +experiment=test cfg_dir=/home/aghinassi/VSR/checkpoints/basic_cl_gan
python -u test_torch.py +experiment=test cfg_dir=/home/aghinassi/VSR/checkpoints/basic_cl_gan2