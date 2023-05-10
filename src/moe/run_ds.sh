#!/bin/bash
export PYTHONPATH=/home/aghinassi/miniconda3/envs/vsr/bin/python
cd /home/aghinassi/Desktop/nn-lab/src

deepspeed --hostfile moe/hostfile \
          --master_addr rackete \
          --master_port 1234 \
          --no_ssh_check \
            moe/cifar10_deepspeed.py \
	        --deepspeed \
	        --deepspeed_config moe/ds_config.json
