#!/bin/bash

set -e -x

# python test.py +experiment=test cfg_dir=../../checkpoints/basic_gan
# python test.py +experiment=test cfg_dir=../../checkpoints/basic_cl_gan
# python test.py +experiment=test cfg_dir=../../checkpoints/basic_cl_gan2
# python test.py +experiment=test cfg_dir=../../checkpoints/basic_spynet
python test.py +experiment=test cfg_dir=../../checkpoints/vrt
# python test.py +experiment=test cfg_dir=../../checkpoints/vrt_cl
# python test.py +experiment=test cfg_dir=../../checkpoints/vrt_spynet
# python test.py +experiment=test cfg_dir=../../checkpoints/moe
# python test.py +experiment=test cfg_dir=../../checkpoints/moe_cl
# python test.py +experiment=test cfg_dir=../../checkpoints/moe_cl2
# python test.py +experiment=test cfg_dir=../../checkpoints/moe_spynet
# python test.py +experiment=test cfg_dir=../../checkpoints/moe_8e
# python test.py +experiment=test cfg_dir=../../checkpoints/moe_8e_cl
# python test.py +experiment=test cfg_dir=../../checkpoints/moe_8e_cl2