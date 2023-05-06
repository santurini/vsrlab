#!/usr/bin/env zsh
git pull

export MASTER_ADDR=192.168.1.42
export MASTER_PORT=1234

echo $MASTER_ADDR
echo $MASTER_PORT

mpirun -v -np 2 --hostfile nodes.txt --map-by node \
        python train_torch.py +experiment=basic_of_ddp \
        core.run_id=mpirun core.run_name=MPIRUN
