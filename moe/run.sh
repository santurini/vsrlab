#!/bin/bash

EP_SIZE=4 # Size of expert parallel world (should be less than total world size)
EXPERTS=4 # Number of total experts per layer
K=2

# ep 2 exp 2 top-k 1 --> 1.8 giga, 54%
# ep 2 exp 2 top-k 2 --> 1.8 giga, 54%
# ep 2 exp 4 top-k 1 --> 1.8 giga, 57%
# ep 2 exp 4 top-k 1 --> 1.8 giga, 54%

deepspeed --hostfile=hostfile \
          --master_addr=192.168.1.42 \
          --master_port=1234 \
          --launcher=openmpi \
          --launcher_args="-bind-to none -map-by slot -x PATH=$PATH:/home/aghinassi/miniconda3/envs/vsr/bin" \
            cifar10_moe.py --deepspeed --deepspeed_config ds_config.json \
          --moe \
          --ep-world-size ${EP_SIZE} \
	        --num-experts-per-layer ${EXPERTS} \
          --top-k ${K} \
	        --noisy-gate-policy 'RSample' \
	        --moe-param-group