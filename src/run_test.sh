#!/bin/bash

set -x

python -u test_torch.py +experiment=test cfg_dir=/home/aghinassi/VSR/checkpoints/vrt_gan
python -u test_torch.py +experiment=test cfg_dir=/home/aghinassi/VSR/checkpoints/vrt_gan_of
python -u test_torch.py +experiment=test cfg_dir=/home/aghinassi/VSR/checkpoints/basic_cl_gan
python -u test_torch.py +experiment=test cfg_dir=/home/aghinassi/VSR/checkpoints/basic_cl_gan2
python -u test_torch.py +experiment=test cfg_dir=/home/aghinassi/VSR/checkpoints/basic_gan_lessadv