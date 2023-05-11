#!/bin/bash

deepspeed --hostfile=hostfile \
          --master_addr=192.168.1.42 \
          --master_port=1234 \
          --launcher=openmpi \
          --launcher_args="-bind-to none -map-by slot -x PATH=$PATH:/home/aghinassi/miniconda3/envs/vsr/bin" \
            train_moe.py --deepspeed --deepspeed_config ds_config.json
