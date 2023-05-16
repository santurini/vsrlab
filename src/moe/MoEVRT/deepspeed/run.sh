#!/bin/bash
PROJECT_ROOT="$HOME/Desktop/nn-lab/src"

deepspeed --hostfile="$PROJECT_ROOT/vsr/models/MoEVRT/hostfile" \
          --master_addr=localhost \
          --master_port=1234 \
          --launcher=openmpi \
          --launcher_args="-bind-to none -map-by slot -x PATH=$PATH:/home/aghinassi/miniconda3/envs/vsr/bin" \
          train_moe.py --deepspeed --deepspeed_config "$PROJECT_ROOT/vsr/models/MoEVRT/ds_config.json" \
          --experiment=vrt_moe
