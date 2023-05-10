#!/bin/bash

deepspeed --master_addr 192.168.1.42 \
          --master_port 1234 \
          --no_ssh_check \
          --launcher OpenMPI \
            moe/cifar10_deepspeed.py \
	        --log-interval 100 \
	        --deepspeed \
	        --deepspeed_config moe/ds_config.json
