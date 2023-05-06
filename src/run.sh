#!/bin/zsh

git pull

export MASTER_ADDR=192.168.1.42
export MASTER_PORT=1234

conda activate vsr

mpirun -v -np 2 -host 192.168.1.42,192.168.0.166 \
        --rank-by node --map-by node \
        python train_torch.py +experiment=basic_of_ddp core.run_id=mpirun core.run_name=MPIRUN
