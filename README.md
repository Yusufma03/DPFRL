# DPFRL
The PyTorch implementation of DPFRL: 

Xiao Ma, Peter Karkus, David Hsu, Wee Sun Lee, Nan Ye: [Discriminative Particle Filter Reinforcement Learning for Complex Partial Observations](https://openreview.net/forum?id=HJl8_eHYvS). International Conference on Learning Representations (ICLR), 2020

## Setup the Environment
You can choose either to use Docker or install dependencies yourself. I strongly recommend you to use Docker :)

### Using docker

With `nvidia-docker` installed, first create the container:
```
cd docker
./build.sh
```
which builds the docker container (this will take a few minutes). Once that is done, you can run experiments from the main folder in a container using
```
cd ..
./docker/run.sh <gpu-nr> <name> <command>
```
for example
```
./docker/run.sh 0 atari ./code/main.py -p with environment.config_file=openaiEnv.yaml
```

### Installing dependencies
You will need
- Python v3.6.3 (I used Anaconda environment. Please do not use Python v3.7.x. I had problems during the compilation of mpi4py, which makes it unable to install the corresponding openai baselines)
- [Pytorch](https://pytorch.org/) > v0.4.x
- `conda install -c anaconda mpi4py`
- [openai baselines](https://github.com/openai/baselines). I used an older version of openai baselines, to install, please use: `pip install -e git+https://github.com/openai/baselines.git@bd390c2adecb5d606c455c9fd7099b674add3109#egg=baselines`
- `pip install 'gym[atari]'==0.10.9`

As well as other dependencies by running
```
pip install -r requirements.txt
```
in the main folder.
If you're running into an error with matplotlib on MacOS when running the RNN on MountainHike, you can use [this simple solution](https://stackoverflow.com/a/21789908/3730984).

### Download the data
To test on the Natural Flickering Atari games benchmark, please first download the data [here](https://drive.google.com/drive/folders/1L2bFzwSzhjs540RHDVh8uVKsd7BiJxgM?usp=sharing), and put it at the root of your folder.

## How to use the code

### Flickering Atari Games

The default configuration can be found in `code/conf/` in the `default.yaml`.
The environment must be specified in the command line by `environment.config_file='<envName>.yaml'`. The corresponding yaml file will be loaded as well (and overwrites some values in `default.yaml`, like for example the encoder/decoder architecture to match the observations space).
Everything specified additionally in the command line overwrites the values in both yaml files.

DPFRL:
```
python ./code/main.py -p with environment.config_file=openaiEnv.yaml environment.name=PongNoFrameskip-v0 algorithm.model.h_dim=256 algorithm.multiplier_backprop_length=10 loss_function.num_frames=25.0e06 opt.lr=2.0e-04 algorithm.model.num_particles=15 algorithm.model.particle_aggregation=mgf environment.noise_type=blank
```
(or with other Atari environment)

To test the performance of each model on the Atari variants, use
```
environment.noise_type=blank_back
```

### Mountain Hike
To test the MountainHike task,

DPFRL:
```
python ./code/main.py -p with environment.config_file=mountainHike.yaml algorithm.model.h_dim=128 algorithm.model_detach_encoder=True algorithm.model.num_particles=30 loss_function.num_frames=25.0e06 opt.lr=1.0e-04 algorithm.model.particle_aggregation=mgf environment.config.noise_length=100 environment.config.observation_std=0.1
```


To control the length of the noise vector in Mountain Hike, use
```
environment.config.noise_length=<length>
```

## Logging

Note that we have applied reward clipping to stablize the training, so the logged reward during training might not corresponds to the true reward.

The true results are saved in `saved_runs/<run_id>/metrics.json`, together with the configuration files and the saved models. For easy visualization, we also log it using tensorboard. Please use `tensorboard --logdir ./tfboard_runs/<run id>`.

## Cite DPFRL
If you find this work useful, please consider citing us
```
@inproceedings{
ma2020discriminative,
title={Discriminative Particle Filter Reinforcement Learning for Complex Partial observations},
author={Xiao Ma and Peter Karkus and David Hsu and Wee Sun Lee and Nan Ye},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=HJl8_eHYvS}
}

```

## Credits

The code is based on an older version of [DVRL's PyTorch implementation](https://github.com/maximilianigl/DVRL), but heavily modified. The PFGRU model is adapted from [PF-RNN's PyTorch implementation](https://github.com/Yusufma03/pfrnns). Please also consider citing them.

```
@article{igl2018deep,
  title={Deep variational reinforcement learning for pomdps},
  author={Igl, Maximilian and Zintgraf, Luisa and Le, Tuan Anh and Wood, Frank and Whiteson, Shimon},
  journal={arXiv preprint arXiv:1806.02426},
  year={2018}
}
```

```
@article{ma2019particle,
  title={Particle Filter Recurrent Neural Networks},
  author={Ma, Xiao and Karkus, Peter and Hsu, David and Lee, Wee Sun},
  journal={arXiv preprint arXiv:1905.12885},
  year={2019}
}
```
