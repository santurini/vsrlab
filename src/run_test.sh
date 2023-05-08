#!/bin/bash

python test_torch.py +experiment=test
python test_torch.py +experiment=test cfg_dir=/home/aghinassi/VSR/checkpoints/basic_cl_gan
python test_torch.py +experiment=test cfg_dir=/home/aghinassi/VSR/checkpoints/basic_cl_gan2