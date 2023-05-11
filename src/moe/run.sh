#!/bin/bash

deepspeed --hostfile=moe/hostfile \
          --master_addr=192.168.1.42 \
          --master_port=1234 \
          --launcher=OpenMPI \
          --launcher_args='-bind-to none -map-by slot -x MASTER_PORT=1234 ' \
          moe/train.py --deepspeed --deepspeed_config "$(realpath moe/ds_config.json)"
