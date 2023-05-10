#!/bin/bash

deepspeed --master_addr localhost \
          --master_port 1234 \
          --no_ssh_check \
          --launcher OpenMPI \
            moe/cifar10_deepspeed.py \
	        --log-interval 100 \
	        --deepspeed \
	        --deepspeed_config moe/ds_config.json
