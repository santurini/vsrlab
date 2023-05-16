#!/bin/bash
PROJECT_ROOT="$HOME/Desktop/nn-lab/src/moe/MoEVRT"

export CC=gcc-10
export CXX=g++-10

deepspeed --hostfile="$PROJECT_ROOT/deepspeed/hostfile" \
          --master_addr=10.23.0.80 \
          --master_port=1234 \
          --launcher=openmpi \
          --launcher_args="-bind-to none -map-by slot -x PATH=$PATH:/home/aghinassi/venv/torch/bin" \
          train_moe.py --deepspeed --deepspeed_config "$PROJECT_ROOT/deepspeed/ds_config.json" \
          --experiment=vrt_moe
