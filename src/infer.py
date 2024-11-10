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
    # determine the environment
    env = Env(LUNAR_LANDER, 'human')
    # create the model
    model = DQN(*env.get_space())
    # initalize the agent
    agent = Agent(model, weights=model.get_filename())
    # run an episode
    reward = run_episode(env, agent)
    print('episode: 0', 'reward:', reward)
    pass


if __name__ == "__main__":
    main()
