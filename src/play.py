# Play as the lunar lander within the environment using the keyboard.

import argparse
import gymnasium as gym
from gymnasium.utils.play import play
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


class RewardWrapper(gym.Wrapper):

    def reset(self, **kwargs):
        # Call the original reset function
        observation, info = super().reset(**kwargs)

        self.episode_reward = 0
        self.episode_length = 0
        
        return observation, info
    
    def step(self, action):
        # Call the original step function
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
    

def play_discrete():
    play(RewardWrapper(gym.make("LunarLander-v3", continuous=False, render_mode="rgb_array")),  
         keys_to_action={
             " ": 2,
             "w": 2,
             "a": 1,
             "d": 3,
             },
         noop=0
)


def play_continuous():
    play(RewardWrapper(gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")),
         keys_to_action={
             " ": np.array([1, 0], dtype=np.float32),
             "w": np.array([1, 0], dtype=np.float32),
             "a": np.array([-1, -1], dtype=np.float32),
             "d": np.array([-1, 1], dtype=np.float32),
             "a ": np.array([1, -1], dtype=np.float32),
             "d ": np.array([1, 1], dtype=np.float32),
             "aw": np.array([1, -1], dtype=np.float32),
             "dw": np.array([1, 1], dtype=np.float32),
             },
         noop=np.array([-1, 0], dtype=np.float32)
    )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("action_space", type=str, help="continuous or discrete")
    args = parser.parse_args()

    if args.action_space == 'continuous':
        play_continuous()
    elif args.action_space == 'discrete':
        play_discrete()
    else:
        print("Please choose either \'continuous\' or \'discrete\' for the action space.")