# Put a model (hopefully already trained) through inference.

import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing

from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from models import PPO
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
import torch
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)

from lib import Env, Agent, LUNAR_LANDER, LUNAR_LANDER_GYM

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

    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )

    env = Env(LUNAR_LANDER, render=None, device=device, continuous=True, use_ppo=True)

    env.env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    rollout = env.env.rollout(3)
    print("rollout of three steps:", rollout)
    print("Shape of the rollout TensorDict:", rollout.batch_size)

    # initialize the agent
    model = PPO(num_cells, 2*env.env.action_spec.shape[-1], device=device)

    # initalize the agent
    agent = Agent(model, weights='weights/ppo_8x256x256x256x4.pth')

    policy_module = TensorDictModule(
        agent.model.net, in_keys=["observation"], out_keys=["loc", "scale"]
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.env.action_spec.space.low,
            "max": env.env.action_spec.space.high,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )

    policy_module(env.reset())

    rewards = []
    avg = 0.0

    TRIALS = 2

    # create the model
    for i in range(TRIALS):
        env.reset()
        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.env.rollout(1000, policy_module)
            print('eval reward:', eval_rollout["next", "reward"].mean().item())
            print('eval reward (sum):', eval_rollout["next", "reward"].sum().item())
            print('eval step_count:', eval_rollout["step_count"].max().item())
            rewards += [eval_rollout['next', 'reward'].sum().item()]
            avg += rewards[-1]
            del eval_rollout
        pass

    print('average:', avg/TRIALS)

    with open("output/ppo_rewards.txt", 'w') as fd:
        for i in rewards:
            fd.write(str(i) + '\n')
    pass


if __name__ == "__main__":
    main()
