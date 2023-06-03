#!/bin/bash

set -e -x

# python test.py +experiment=test cfg_dir=../../checkpoints/basic_gan
# python test.py +experiment=test cfg_dir=../../checkpoints/basic_cl_gan
# python test.py +experiment=test cfg_dir=../../checkpoints/basic_cl_gan2
# python test.py +experiment=test cfg_dir=../../checkpoints/basic_spynet
# python test.py +experiment=test cfg_dir=../../checkpoints/vrt
# python test.py +experiment=test cfg_dir=../../checkpoints/vrt_easy
# python test.py +experiment=test cfg_dir=../../checkpoints/vrt_spynet
# python test.py +experiment=test cfg_dir=../../checkpoints/moe
# python test.py +experiment=test cfg_dir=../../checkpoints/moe_easy
# python test.py +experiment=test cfg_dir=../../checkpoints/moe_easy2
# python test.py +experiment=test cfg_dir=../../checkpoints/moe_spynet
# python test.py +experiment=test cfg_dir=../../checkpoints/moe_8e
python test.py +experiment=test cfg_dir=../../checkpoints/moe_8e_easy
python test.py +experiment=test cfg_dir=../../checkpoints/moe_8e_invcl

# TODO: vrt_cl, moe_cl, moe_8e_cl, moe_spynet2