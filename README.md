# Video Super Resolution Playground
A library to train, test and develop Video Super Resolution architectures.
The framework is based on [Hydra](https://github.com/facebookresearch/hydra) and offers a variety of SoA architectures ([Real Basic VSR](https://arxiv.org/abs/2111.12704), [VRT](https://arxiv.org/abs/2201.12288)) already implemented and ready to be trained.

### Create venv
```
conda create -n vsrlab python=3.11
conda activate vsrlab

pip3 install -r requirements.txt
```

##### Install the package locally
```
git clone https://github.com/santurini/vsrlab.git

cd vsrlab && pip3 install .
```

### Quick examples
```
python train.py +experiment=basic
# to resume a training -> change 'restore' and 'restore_opt' fields in experiment config file 

python test.py +experiment=test cfg_dir=path/to/checkpoints_dir/experiment_name
# checkpoints directory should contain 'checkpoint/last.ckpt' and 'config.yaml'
```
