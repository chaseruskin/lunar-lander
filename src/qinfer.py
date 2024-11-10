import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


env = gym.make("LunarLander-v3", render_mode='human')

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        # create a feed-forward network that takes in the difference between
        # the current and previous screen patches
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    pass


# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
n_observations = env.observation_space.shape[0]

model = DQN(n_observations, n_actions).to(device)

model.load_state_dict(torch.load('checkpoint.pth'))

model.eval()

agent = model

with torch.no_grad():
    episode_over = False
    t_reward = 0
    # Initialize the environment and get its state
    obs, info = env.reset()
    while not episode_over:
        state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = agent(state).argmax().item()
            
        obs, reward, terminated, truncated, info = env.step(action)
        t_reward += reward
        episode_over = terminated or truncated
        if episode_over == True:
            print('episode: 0', 'reward:', t_reward)
            pass
        pass
