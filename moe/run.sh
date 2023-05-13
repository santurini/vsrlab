#!/bin/bash

# Size of expert parallel world (should be less than total world size)
EP_SIZE=2

# Number of total experts per layer
EXPERTS=2

# Number of total expert layers
EXPERT_LAYERS=2

deepspeed --hostfile=hostfile \
          --master_addr=192.168.1.42 \
          --master_port=1234 \
          --launcher=openmpi \
          --launcher_args="-bind-to none -map-by slot -x PATH=$PATH:/home/aghinassi/miniconda3/envs/vsr/bin" \
            train_moe.py --deepspeed --deepspeed_config ds_config.json \
          --moe \
          --ep-world-size ${EP_SIZE} \
	        --num-experts-per-layer ${EXPERTS} \
          --num-expert-layers ${EXPERT_LAYERS} \
          --top-k 1 \
	        --noisy-gate-policy 'RSample' \
	        --moe-param-group