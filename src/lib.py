# Supportive classes for RL; namely environments and agents.

import gymnasium as gym
import torch
from models import Model
from lunar_lander import LunarLander
from torchrl.envs.libs.gym import GymEnv

from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)

# The custom environment
LUNAR_LANDER: str = 'cse592/LunarLander'

# The old environment (directly from `gymnasium`)
LUNAR_LANDER_GYM: str = 'LunarLander-v3'

class Env:

    _initialized = False

    def __init__(self, name: str, render: str=None, continuous: bool=False, device: str=None, use_ppo: bool=False):
        """
        Configure an enviroment through Gymnasium.
        """
        Env.register_custom()
        self.env = gym.make(name, render_mode=render, continuous=continuous)
        if use_ppo == True:
            ppo_env = GymEnv(name, render_mode=render, device=device, continuous=continuous)
            transform_env = TransformedEnv(
                ppo_env,
                Compose(
                    # normalize observations
                    ObservationNorm(in_keys=["observation"]),
                    DoubleToFloat(),
                    StepCounter(),
                ),
            )
            self.env = transform_env
        pass

    def get_space(self):
        """
        Returns the number of observations (state space) and the number of actions
        (action space).
        """
        return (self.env.observation_space.shape[0], self.env.action_space.n)
    
    def reset(self, gravity: float=None):
        """
        Initialize the environment.
        """
        return self.env.reset(options={'gravity': gravity})
    
    def step(self, action):
        """
        Update the environment based on the agent's `action`.
        """
        return self.env.step(action)
    
    @staticmethod
    def register_custom():
        if Env._initialized == False:
            # register our custom environment
            gym.register(
                id=LUNAR_LANDER,
                entry_point=LunarLander,
            )
            Env._initialized = True
        pass


class Agent:

    def __init__(self, model: Model, weights=None, device=None):
        """
        Creates an agent with the given `model`.
     
        ### Args
        - `model`: the torch model to use
        - `training`: if false, sets the model to eval mode
        - `weights`: filepath to weights to load from memory
        - `device`: manually choose where to run the model and its computations
        """
        if device == None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else
                "cpu"
            )
        self.model = model
        self.device = device
        self.model.to(self.device)
        # load weights from a file
        if weights != None:
            print('info: loading weights from file:', weights)
            self.model.load_state_dict(torch.load(weights, map_location=torch.device(self.device)))
        # set the model to evaluation mode
        self.model.eval()
        pass

    def select_action(self, state):
        """
        Select the next best action according the agent's policy.
        """
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        # print(self.model, self.model(state))
        sel = self.model(state)
        if type(sel) == tuple:
            sel = sel[0]
        action = sel.argmax().item()
        return action
