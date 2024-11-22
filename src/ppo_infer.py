# Put a model (hopefully already trained) through inference.

import warnings
warnings.filterwarnings("ignore")

from models import PPO
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
import torch
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)

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
    num_cells = 256
    # determine the environment
    env = Env(LUNAR_LANDER, 'human')

    spec = TransformedEnv(
        env.env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    ).action_spec

    # create the model
    model = PPO(num_cells, *env.get_space())
    # initalize the agent
    agent = Agent(model, weights='weights/ppo_8x256x256x256x4.pth')

    policy_module = TensorDictModule(
        model.net, in_keys=["observation"], out_keys=["loc", "scale"]
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": 0,
            "max": spec.n,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )
    
    # run an episode
    rewards = []
    avg = 0.0
    TRIALS = 2
    for i in range(TRIALS):
        reward = run_episode(env, agent)
        rewards += [reward]
        print('episode:', i, 'reward:', reward)
        avg += reward

    print('average:', avg/TRIALS)


    with open("output/ppo_rewards.txt", 'w') as fd:
        for i in rewards:
            fd.write(str(i) + '\n')
    pass


if __name__ == "__main__":
    main()
