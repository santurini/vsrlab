#!/bin/bash

torchrun  --nnodes 2 \
          --master_addr 192.168.1.42 \
          --master_port 1234 \
          python moe/cifar10_deepspeed.py \
	        --log-interval 100 \
	        --deepspeed \
	        --deepspeed_config moe/ds_config.json
