#!/bin/bash

deepspeed --hostfile moe/hostfile \
            moe/cifar10_deepspeed.py \
	        --deepspeed \
	        --deepspeed_config moe/ds_config.json
