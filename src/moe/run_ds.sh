#!/bin/bash

deepspeed --hostfile moe/hostfile \
          --no_ssh_check \
            moe/cifar10_deepspeed.py \
	        --deepspeed \
	        --deepspeed_config moe/ds_config.json
