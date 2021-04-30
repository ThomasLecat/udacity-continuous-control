# udacity-continuous-control-project

This repository implements an RL agent that solves the Unity reacher environment.

The agent is DDPG.

## Installation

This project uses the drlnd conda environment from the Udacity Deep Reinforcement
Learning program.

1. Follow the instructions from Udacity's [README](https://github.com/udacity/deep-reinforcement-learning#dependencies) 
to create the environment and install the dependencies.
1. Install the project's package: `$ source activate drlnd && pip install -e .`
1. Download the RL environment for your OS, place the file in the `ccontrol/` directory 
and unzip (or decompress) it. 

*  Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Reacher/Reacher_Linux.zip)
*  Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Reacher/Reacher.app.zip)
*  Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Reacher/Reacher_Windows_x86.zip)
*  Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Reacher/Reacher_Windows_x86_64.zip)

*(Optional)* To contribute, install the pre-commits:

```bash
$ pre-commit install
```

## Usage

Before training or evaluating an agent, make sure you conda environment is activated:
```
$ source activate drlnd
```

### Training

1. Tune DDPG's learning parameters in `ccontrol/config.py`
2. run `python ccontrol/train.py --environment_path /path/to/Reacher.app`. You can 
also specify the number of training episodes with the `--num_episodes` argument.

At the end of training, two files are saved on disk:
*  `dqn_checkpoint.pt`: PyTorch checkpoint containing the trained model's weights.
*  `reward_per_episode.csv`: score of all training episodes.

### Evaluation

Using the same config parameters as in training, run:
```
python ccontrol/evaluate.py --environment_path /path/to/Reacher.app --checkpoint_path dqn_checkpoint.pt --show_graphics True
```

## Description of the environment

In this environment, a double-jointed arm can move to target locations. 

A reward of +0.1 is provided for each step that the agent's hand is in the goal 
location. The goal is to maintain its position at the target location for as many time 
steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, 
velocity, and angular velocities of the arm. 

The environment has a discrete action space: each action is a vector with four numbers, 
corresponding to torque applicable to two joints. Every entry in the action vector 
should be a number between -1 and 1.

The environment is considered solved when the agent receives an average reward of at 
least +30 over 100 consecutive episodes.

![Agent playing on Reacher environment](doc/reacher.gif)
