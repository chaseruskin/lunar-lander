# Lunar Lander

#### Authors: Daniel Skora, Nihal Simha, Chase Ruskin

Final project for University of Michigan's CSE592: Foundations of AI. The goal of this project is to apply reinforcement learning (RL) methodologies to solving the lunar landing problem for generalized environments.

## Project Organization

The project is organized as follows:

- `/src`: Python scripts for RL experimentation (tracked)
- `/output`: Run-dependent artifacts that (untracked)
- `/data`: Plots and statistics regarding our experiments (tracked)
- `/weights`: Saved weight parameters from trained networks (tracked)

## Getting Started

This project has been tested using Python >= 3.8.6. 

1. Install the Python-related dependencies:

```
pip install -r requirements.txt
```

2. Run our experiment:

```
python src/main.py
```

## Usage

To train the DQN model:

```
python src/learn.py
```

To train the PPO model:

```
python src/ppo_learn.py
```

To see the trained DQN model run inference:

```
python src/infer.py
```

To see the trained PPO model run inference:

```
python src/ppo_infer.py
```

To manually maneuver the lander using the keyboard:

```
python src/play.py [continuous, discrete]
```

## References

- https://gymnasium.farama.org
- https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html