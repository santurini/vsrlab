# VSR Playground

#### Create VE
```
conda create -n vsrlab python=3.11
conda activate vsrlab

pip3 install -r requirements.txt
```

#### Download FastMoE
Go at this [link](https://github.com/laekov/fastmoe/blob/master/doc/installation-guide.md) and follow the instructions.

#### Install the package locally
```
git clone https://gitlab.com/santurini/vsrlab.git

cd vsrlab && pip3 install .
```

#### Quick test example
```
python train.py +experiment=moevrt
# to resume a training -> change 'restore' and 'restore_opt' fields in experiment config file 

python test.py +experiment=test cfg_dir=path/to/checkpoints_dir/experiment_name
# checkpoints directory should contain 'checkpoint/last.ckpt' and 'config.yaml'
```
