#!/bin/bash

git pull
deepspeed --hostfile=moe/hostfile \
          --launcher=OpenMPI \
          --launcher_args='-x MASTER_ADDR=192.168.1.42 -x MASTER_PORT=1234 -x PATH=/home/aghinassi/miniconda3/envs/vsr/bin -bind-to none -map-by slot' \
          moe/train.py --deepspeed --deepspeed_config ./ds_config.json
