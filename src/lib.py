import gymnasium as gym
import torch
from models import Model

LUNAR_LANDER: str = 'LunarLander-v3'

class Env:

    def __init__(self, name: str, render: str=None):
        """
        Configure an enviroment through Gymnasium.
        """
        self.env = gym.make(name, render_mode=render)

    def get_space(self):
        """
        Returns the number of observations (state space) and the number of actions
        (action space).
        """
        return (self.env.observation_space.shape[0], self.env.action_space.n)
    
    def reset(self):
        """
        Initialize the environment.
        """
        return self.env.reset()
    
    def step(self, action):
        """
        Update the environment based on the agent's `action`.
        """
        return self.env.step(action)


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
            self.model.load_state_dict(torch.load(weights))
        # set the model to evaluation mode
        self.model.eval()
        pass

    def select_action(self, state):
        """
        Select the next best action according the agent's policy.
        """
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.model(state).argmax().item()
        return action

    pass