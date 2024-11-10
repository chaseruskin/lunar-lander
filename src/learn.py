import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

from models import Model, DQN
from copy import copy
from lib import Env, LUNAR_LANDER

import torch
import torch.nn as nn
import torch.optim as optim

steps_done = 0
episode_durations = []

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


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

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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


def select_action(state, env: Env, policy: Model, hp: HyperParameters, device: str):
    global steps_done
    sample = random.random()
    eps_threshold = hp.EPS_END + (hp.EPS_START - hp.EPS_END) * \
        math.exp(-1. * steps_done / hp.EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.env.action_space.sample()]], device=device, dtype=torch.long)


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model(optimizer, policy: Model, target: Model, memory: ReplayMemory, hp: HyperParameters, device: str):
    if len(memory) < hp.BATCH_SIZE:
        return
    transitions = memory.sample(hp.BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(hp.BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * hp.GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy.parameters(), 100)
    optimizer.step()


def go_train(env: Env, policy: Model, target: Model, hp: HyperParameters, num_episodes: int, device: str):

    optimizer = optim.AdamW(policy.parameters(), lr=hp.LR, amsgrad=True)
    memory = ReplayMemory(10000)

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        
        i_reward = 0
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = select_action(state, env, policy, hp, device)
            observation, reward, terminated, truncated, info = env.step(action.item())
            i_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(optimizer, policy, target, memory, hp, device)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target.state_dict()
            policy_net_state_dict = policy.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*hp.TAU + target_net_state_dict[key]*(1-hp.TAU)
            target.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break
        print('episode:', i_episode, 'reward:', i_reward)
        pass
    return policy



def train(env, model: Model, hp: HyperParameters, episodes: int):
    """
    Train the model.

    ### Args
    - `env`: the environment to train in
    - `model`: the model to be trained
    - `hp`: the set of hyperparameters
    - `episodes`: the total number of episodes to run
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    print('info: training on device:', device)
    
    policy = copy(model).to(device)
    target = copy(model).to(device)
    target.load_state_dict(policy.state_dict())

    policy_after_training = go_train(env, policy, target, hp, episodes, device)

    torch.save(policy_after_training.state_dict(), policy_after_training.get_filename())

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()


def main():
    # determine how long we should train for
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        episodes = 600
    else:
        episodes = 10
    # configure the hyper parameters for training
    hp = HyperParameters()
    # initialize the environment
    env = Env(LUNAR_LANDER)
    # create the model we wish to train
    model = DQN(*env.get_space())
    # train the model!
    train(env, model, hp, episodes)
    pass


if __name__ == '__main__':
    main()