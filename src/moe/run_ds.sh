#!/bin/bash

deepspeed --num_nodes 2 \
          --num_gpus 1 \
          --hostfile moe/hostfile \
          --master_addr 192.168.1.42 \
          --master_port 1234 \
            moe/cifar10_deepspeed.py \
	        --deepspeed \
	        --deepspeed_config moe/ds_config.json
