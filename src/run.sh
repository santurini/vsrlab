#!/bin/bash

mpirun -np 2 \
-H 192.168.0.166:1,192.168.1.42:1 \
-x MASTER_ADDR=192.168.0.166 \
-x MASTER_PORT=1234 \
-bind-to none -map-by node \
python train_torch.py +experiment=baby_of core.run_id="sanity" core.run_name="Sanity Checking"