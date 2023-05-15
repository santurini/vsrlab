#!/bin/bash
PROJECT_ROOT="$HOME/Desktop/nn-lab/src/moe/MoEVRT"
export TORCH_DISTRIBUTED_DEBUG=OFF

deepspeed --hostfile="$PROJECT_ROOT/deepspeed/hostfile" \
          --master_addr=192.168.1.42 \
          --master_port=1234 \
          --launcher=openmpi \
          --launcher_args="-bind-to none -map-by slot -x PATH=$PATH:/home/aghinassi/miniconda3/envs/vsr/bin" \
          train_moe.py --deepspeed --deepspeed_config "$PROJECT_ROOT/deepspeed/ds_config.json" \
          --experiment=vrt_moe
