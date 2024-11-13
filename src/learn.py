# Train a model.

import math
import time
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from copy import copy

import torch
import torch.nn as nn
import torch.optim as optim

from models import Model, DQN
from lib import Env, LUNAR_LANDER

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class HyperParameters:

    def __init__(self, 
                 batch_size=128, 
                 gamma=0.99, 
                 eps_start=0.9, 
                 eps_end=0.05, 
                 eps_decay=1_000, 
                 tau=0.005, 
                 lr=1e-4):
        """
        ### Args
        - `BATCH_SIZE` is the number of transitions sampled from the replay buffer
        - `GAMMA` is the discount factor as mentioned in the previous section
        - `EPS_START` is the starting value of epsilon
        - `EPS_END` is the final value of epsilon
        - `EPS_DECAY` controls the rate of exponential decay of epsilon, higher means a slower decay
        - `TAU` is the update rate of the target network
        - `LR` is the learning rate of the ``AdamW`` optimizer
        """
        # store the set of parameters in their own variables
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TAU = tau
        self.LR = lr
    
    pass


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """
        Save a transition
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Trainer:

    def __init__(self, env: Env, model: Model, hp: HyperParameters, visualize: bool=True):
        """
        Initialize a trainer for the model in the particular environment.

        ### Args
        - `env`: the environment to train in
        - `model`: the model to be trained
        - `hp`: the set of hyperparameters
        - `episodes`: the total number of episodes to run
        - `visualize`: choose whether to plot the data as it trains
        """
        if visualize == True:
            plt.ion()

        self.hp = hp
        self.visualize = visualize
        self.env = env

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        self.steps_done = 0
        self.episode_durations = []
        self.episode_rewards = []

        print('info: training on device:', self.device)
        
        self.policy = copy(model).to(self.device)
        self.target = copy(model).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        pass

    def train(self, n: int):
        """
        Undergo training for `n` episodes.
        """

        optimizer = optim.AdamW(self.policy.parameters(), lr=self.hp.LR, amsgrad=True)
        memory = ReplayMemory(10_000)

        for i_episode in range(n):
            # Initialize the environment and get its state
            state, info = self.env.reset()
            
            i_reward = 0
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, info = self.env.step(action.item())
                i_reward += reward
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model(optimizer, memory)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target.state_dict()
                policy_net_state_dict = self.policy.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.hp.TAU + target_net_state_dict[key]*(1-self.hp.TAU)
                self.target.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    self.episode_rewards.append(i_reward)
                    if self.visualize == True:
                        self.plot_durations()
                        self.plot_rewards()
                    break
            print('episode:', i_episode, 'reward:', i_reward)
            pass
        return self.policy

    def select_action(self, state):
        """
        Choose the next action during within the current training episode.
        """
        sample = random.random()
        eps_threshold = self.hp.EPS_END + (self.hp.EPS_START - self.hp.EPS_END) * \
            math.exp(-1. * self.steps_done / self.hp.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.env.action_space.sample()]], device=self.device, dtype=torch.long)
        pass

    def optimize_model(self, optimizer, memory: ReplayMemory):
        if len(memory) < self.hp.BATCH_SIZE:
            return
        transitions = memory.sample(self.hp.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.hp.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.hp.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        optimizer.step()

    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.grid(True)
        plt.legend(['Duration', '100 episode average'])
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if show_result == True:
            if self.visualize == True:
                plt.ioff()
            plt.show()
        
    def plot_rewards(self, show_result=False):
        plt.figure(2)
        rewards_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.legend(['Reward', '100 episode average'])
        plt.plot(rewards_t.numpy())
        # Take 100 episode averages and plot them too
        if len(rewards_t) >= 100:
            means_r = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means_r = torch.cat((torch.zeros(99), means_r))
            plt.plot(means_r.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if show_result == True:
            if self.visualize == True:
                plt.ioff()
            plt.show()

    pass


def main():    
    # determine how long we should train for
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        episodes = 800
    else:
        episodes = 10

    # configure the hyper parameters for training
    hp = HyperParameters()
    # initialize the environment
    env = Env(LUNAR_LANDER)
    # create the model we wish to train
    model = DQN(*env.get_space())

    # set up the trainer
    trainer = Trainer(env, model, hp, visualize=True)

    # train the model!
    start = time.time()
    policy = trainer.train(episodes)
    print('info: training finished')
    print('info: time elapsed for training:', str(time.time() - start), 's')  

    # save the weights
    policy.save()
    print('info: weights saved to file:', '\"' + str(policy.get_filename()) + '\"')

    trainer.plot_durations(show_result=True)
    trainer.plot_rewards(show_result=True)
    pass


if __name__ == '__main__':
    main()