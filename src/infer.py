# Put a model (hopefully already trained) through inference.

from models import DQN
import torch
from lib import Env, Agent, LUNAR_LANDER

def run_episode(env: Env, agent: Agent):
    """
    Runs a single episode for the agent `agent` interacting within the environment `env`.
    """
    episode_over = False
    # record data about the episode over its duration
    t_reward = 0

    obs, info = env.reset()
    with torch.no_grad():
        while not episode_over:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            # accumulate the reward over the episode's duration
            t_reward += reward
            episode_over = terminated or truncated
    return t_reward


def main():
    render = 'human'
    # determine the environment
    env = Env(LUNAR_LANDER, render)
    # create the model
    model = DQN(*env.get_space())
    # initalize the agent
    agent = Agent(model, weights='weights/dqn_8x256x256x256x4_gunifrom_10_6_600.pth')
    # run an episode
    rewards = []
    avg = 0.0
    TRIALS = 3
    for i in range(TRIALS):
        reward = run_episode(env, agent)
        rewards += [reward]
        print('episode:', i, 'reward:', reward)
        avg += reward

    print('average:', avg/TRIALS)

    with open("output/rewards.txt", 'w') as fd:
        for i in rewards:
            fd.write(str(i) + '\n')
    pass


if __name__ == "__main__":
    main()
