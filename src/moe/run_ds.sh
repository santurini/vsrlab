#!/bin/bash

# Number of nodes
NUM_NODES=2
# Number of GPUs per node
NUM_GPUS=1

deepspeed --hostfile /Users/arturo/Desktop/nn-lab/src/moe/hostfile \
          --master_addr 192.168.1.42 \
          --master_port 1234 \
          --launcher OpenMPI \
            cifar10_deepspeed.py \
	        --log-interval 100 \
	        --deepspeed \
	        --deepspeed_config ./ds_config.json
