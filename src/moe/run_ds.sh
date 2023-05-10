#!/bin/bash

deepspeed --hostfile hostfile \
            cifar10_deepspeed.py \
	        --deepspeed \
	        --deepspeed_config ds_config.json
