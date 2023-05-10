#!/bin/bash

# Number of nodes
NUM_NODES=2
# Number of GPUs per node
NUM_GPUS=1

deepspeed --master_addr 192.168.1.42 \
          --master_port 1234 \
          --launcher OpenMPI \
            moe/cifar10_deepspeed.py \
	        --log-interval 100 \
	        --deepspeed \
	        --deepspeed_config moe/ds_config.json
