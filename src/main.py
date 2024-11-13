# Entry script for the lunar lander project.

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from gymnasium.utils.play import play
import shutil
import numpy as np

def extract_pos(obs):
    """
    Collect the `x` and `y` of the lunar lander's position from the
    observation space for the lunar lander.
    """
    return np.array([obs[0], obs[1]])


def compute_err(location, goal=np.zeros(2)):
    """
    Compute the normalized error between the final position (`location`) and a
    goal position (`goal`).

    The `location` should be a np.array with 2 elements (x and y).
    """
    return np.linalg.norm(location - goal)


def clean_output(dir='output'):
    """
    Removes all contents from the output folder before the next run.
    """
    shutil.rmtree(dir)
    pass


def train(env):
    """
    Trains the agent for the lunar lander environment.
    """
    # TODO: implement
    pass


def eval(env, agent=None, episodes=1, record=True):
    """
    Evaluate the agent's performance for the given environment.
    """
    if record == True:
        env = RecordVideo(env, video_folder="output", name_prefix="eval",
            episode_trigger=lambda x: True)
    env = RecordEpisodeStatistics(env, buffer_length=episodes)

    for _ in range(episodes):
        obs, info = env.reset()
        print('start pos:', extract_pos(obs))

        episode_over = False
        while not episode_over:
            if agent == None:
                # let a random action occur
                action = env.action_space.sample()
            else:
                # TODO implement agent behavior
                pass
            obs, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated
            if episode_over == True:
                final_location = extract_pos(obs)
                final_vels = (obs[2], obs[3], obs[4])
                print('landed pos:', extract_pos(obs))
                print('final vels:', final_vels)
                print("legs contact:", (obs[6], obs[7]))
                print("dist err:", compute_err(final_location))
                pass
            pass
    env.close()

    # returns (time taken, total rewards, lengths)
    return (env.time_queue, env.return_queue, env.length_queue)

class RewardWrapper(gym.Wrapper):

    def reset(self, **kwargs):
        # Call the original reset function and get the observation and info
        observation, info = super().reset(**kwargs)

        self.episode_reward = 0
        self.episode_length = 0
        
        return observation, info
    
    def step(self, action):
        # Call the original step function and get the observation and info
        observation, reward, terminated, truncated, info = super().step(action)
        episode_over = terminated or truncated
        
        self.episode_reward += reward
        self.episode_length += 1

        if episode_over:
            print("Episode Reward: "+ str(self.episode_reward))
            print("Distance From Goal " + str(compute_err(observation[:2])))
            print("Steps Taken: " + str(self.episode_length))
            print()
        
        return observation, reward, terminated, truncated, info

def human_input():
    play(RewardWrapper(gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")),
    keys_to_action={
        " ": np.array([1, 0], dtype=np.float32),
        "a": np.array([-1, -1], dtype=np.float32),
        "d": np.array([-1, 1], dtype=np.float32),
        "a ": np.array([1, -1], dtype=np.float32),
        "d ": np.array([1, 1], dtype=np.float32),
    },
    noop=np.array([-1, 0], dtype=np.float32)
)

def main():
    """
    Create the environment, run trials, and record results.
    """

    clean_output()

    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                enable_wind=False, wind_power=0.0, turbulence_power=0.0,
                render_mode='rgb_array')
    
    (time, rewards, length) = eval(env, agent=None, episodes=1)

    print(f'Episode time taken: {time}')
    print(f'Episode total rewards: {rewards}')
    print(f'Episode lengths: {length}')
    pass


if __name__ == '__main__':
    # main()

    # DANIEL EXAMPLE
    human_input()