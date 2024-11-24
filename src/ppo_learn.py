import warnings
warnings.filterwarnings("ignore")

import torch.nn.functional as F
import torch
import torch.nn as nn
from models import PPO
from torch import distributions
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from lib import Env, LUNAR_LANDER

def calculate_returns(rewards, gamma):
    returns = []
    cumulative_reward = 0
    for r in reversed(rewards):
        cumulative_reward = r + gamma*cumulative_reward
        returns.insert(0, cumulative_reward)
    returns = torch.tensor(returns)
    # normalize
    returns = (returns - returns.mean()) / returns.std()
    return returns


def calculate_advantages(returns, values):
    advantages = returns.unsqueeze(1) - values
    # normalize
    advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages


def calculate_surrogate_loss(act_log_prob_old, act_log_prob_new, epsilon, advantages):
    advantages = advantages.detach()

    policy_ratio = (act_log_prob_new - act_log_prob_old).exp()
    try:
        surrogate_loss_1 = policy_ratio.unsqueeze(1) * advantages
    except:
        print(policy_ratio)
        print(advantages)
        exit(101)
    surrogate_loss_2 = torch.clamp(policy_ratio.unsqueeze(1), min=1.0-epsilon, max=1.0+epsilon) * advantages
    surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
    return surrogate_loss


def calculate_losses(surrogate_loss, entropy, entropy_coeff, returns, value_pred):
    entropy_bonus = entropy_coeff * entropy
    policy_loss = -(surrogate_loss + entropy_bonus.unsqueeze(1)).sum()
    value_loss = F.smooth_l1_loss(returns.unsqueeze(1), value_pred).sum()
    return policy_loss, value_loss


class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred


def forward_pass(env: Env, agent, gamma):
    states = []
    actions = []
    act_log_prob = []
    values = []
    rewards = []
    done = False
    episode_reward = 0

    device = 'cpu'

    state, info = env.reset()
    agent.train()
    while done == False:
       
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        states.append(state)
        action_pred, value_pred = agent(state)
        # print(action_pred, value_pred)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        state, reward, done, truncated, info = env.step(action.item())
        # print(action, reward)
        # update lists that record the data of the episode
        actions.append(action)
        act_log_prob.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        episode_reward += reward
        pass
    # concatenate each iteration's item into a single vector/dimension
    states = torch.cat(states)
    actions = torch.cat(actions)
    act_log_prob = torch.cat(act_log_prob)
    values = torch.cat(values).squeeze(-1)
    returns = calculate_returns(rewards, gamma=gamma)
    advantages = calculate_advantages(returns, values)
    return episode_reward, states, actions, act_log_prob, advantages, returns


def update_policy(agent, states, actions, act_log_prob_old, advantages, returns, optimizer, ppo_steps, epsilon, entropy_coeff):
    BATCH_SIZE = 128
    total_policy_loss = 0
    total_value_loss = 0
    actions_log_probability_old = act_log_prob_old.detach()
    actions = actions.detach()
    training_results_dataset = TensorDataset(states, actions, actions_log_probability_old, advantages, returns)
    batch_dataset = DataLoader(training_results_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for _ in range(ppo_steps):
        for batch_idx, (states, actions, actions_log_probability_old, advantages, returns) in enumerate(batch_dataset):
            # get new log prob of actions for all input states
            action_pred, value_pred = agent(states)
            value_pred = value_pred
            action_prob = F.softmax(action_pred, dim=-1)
            probability_distribution_new = distributions.Categorical(action_prob)
            entropy = probability_distribution_new.entropy()
            # estimate new log probabilities using old actions
            actions_log_probability_new = probability_distribution_new.log_prob(actions)
            surrogate_loss = calculate_surrogate_loss(actions_log_probability_old, actions_log_probability_new, epsilon, advantages)
            policy_loss, value_loss = calculate_losses(surrogate_loss, entropy, entropy_coeff, returns, value_pred)
            optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            optimizer.step()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        pass
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps


def plot_train_rewards(train_rewards, reward_threshold):
    plt.figure(figsize=(12, 8))
    plt.plot(train_rewards, label='Training Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Training Reward', fontsize=20)
    plt.hlines(reward_threshold, 0, len(train_rewards), color='y')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_losses(policy_losses, value_losses):
    plt.figure(figsize=(12, 8))
    plt.plot(value_losses, label='Value Losses')
    plt.plot(policy_losses, label='Policy Losses')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


def main():


    # initialize the environment
    env = Env(LUNAR_LANDER)

    # hyperparameters
    MAX_EPISODES = 100
    DISCOUNT_FACTOR = 0.99
    REWARD_THRESHOLD = 475
    PRINT_INTERVAL = 10
    PPO_STEPS = 8
    N_TRIALS = 100
    EPSILON = 0.2
    ENTROPY_COEFFICIENT = 0.01
    HIDDEN_DIMENSIONS = 64
    DROPOUT = 0.2
    LEARNING_RATE = 0.001
    train_rewards = []
    policy_losses = []
    value_losses = []

    actor = PPO(HIDDEN_DIMENSIONS, *env.get_space(), DROPOUT)
    critic = PPO(HIDDEN_DIMENSIONS, *env.get_space(), DROPOUT)

    agent = ActorCritic(actor, critic)

    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    for episode in range(1, MAX_EPISODES+1):
        print('epsiode:', episode)
        train_reward, states, actions, act_log_prob, advantages, returns = forward_pass(
            env,
            agent,
            DISCOUNT_FACTOR
        )

        policy_loss, value_loss = update_policy(
            agent,
            states,
            actions,
            act_log_prob,
            advantages,
            returns,
            optimizer,
            PPO_STEPS,
            EPSILON,
            ENTROPY_COEFFICIENT
        )
        
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)

        train_rewards.append(train_reward)

        pass
    plot_train_rewards(train_rewards, REWARD_THRESHOLD)
    plot_losses(policy_losses, value_losses)
    pass


if __name__ == '__main__':
    main()