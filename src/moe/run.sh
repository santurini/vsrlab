#!/bin/bash

git pull

deepspeed --hostfile=moe/hostfile \
          --master_addr=192.168.1.42 \
          --master_port=1234 \
          --launcher=OpenMPI \
          --launcher_args='-x PATH=/home/aghinassi/miniconda3/envs/vsr/bin -bind-to none -map-by slot' \
          moe/train.py --deepspeed --deepspeed_config ./ds_config.json
