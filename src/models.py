# Define the various models used for training and inference.

import torch.nn as nn
import torch
import torch.nn.functional as F
from tensordict.nn.distributions import NormalParamExtractor

# ==========
# Base Model
# ==========

class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.name = 'model'

    def extract_model_data(self):
        """
        Analyzes the `model` to generate a unique string to describe its
        shape.
        """
        data = '_'
        book = dict(self.named_modules())
        keys = list(book)[1:]
        for k in keys:
            l = book[k]
            try:
                data += str(l.in_features) + 'x'
            except:
                pass
        try:
            data += str(l.out_features)
        except:
            data += '4'
        return data

    def get_name(self):
        """
        Return the name of this model.
        """
        return self.name + self.extract_model_data()
    
    def get_filename(self):
        """
        Returns the file name of the given model.
        """
        return 'weights' + '/' + self.get_name() + "_gunifrom_10_6_600.pth"
    
    def save(self):
        """
        Write the model's weights to a file.
        """
        torch.save(self.state_dict(), self.get_filename())
        pass


# =========
# DQN Model
# =========

class DQN(Model):

    def __init__(self, n_observations: int, n_actions: int):
        """
        Create a feed-forward network that takes in the difference between the
        current and previous screen patches.

        ### Args
        - `n_observations`: number of input features
        - `n_actions`: number of output actions
        """
        super(DQN, self).__init__()
        self.name = 'dqn'

        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        # self.layer4 = nn.Linear(512, 512)
        # self.layer5 = nn.Linear(512, 512)
        self.layer4 = nn.Linear(128, n_actions)
        pass
         
    def forward(self, x):
        """
        Determine the next action from one element or a batch during optimization.

        Returns `tensor([[left0exp,right0exp]...])`.
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        # x = F.relu(self.layer4(x))
        # x = F.relu(self.layer5(x))
        return self.layer4(x)
    
    pass


# =========
# PPO Model
# =========

class PPO(Model):

    def __init__(self, num_cells: int, n_outputs: int, device, is_target: bool=False):
        super(PPO, self).__init__()
        self.name = 'ppo'
        # self.layer1 = nn.Linear(n_observations, 256)
        # self.layer2 = nn.Linear(256, 256)
        # self.layer3 = nn.Linear(256, 256)
        # self.layer4 = nn.Linear(256, n_actions)
        if is_target == False:
            self.net = nn.Sequential(
                nn.LazyLinear(num_cells, device=device),
                nn.Tanh(),
                nn.LazyLinear(num_cells, device=device),
                nn.Tanh(),
                nn.LazyLinear(num_cells, device=device),
                nn.Tanh(),
                nn.LazyLinear(n_outputs, device=device),
                NormalParamExtractor(),
            )
        else:
            self.net = nn.Sequential(
                nn.LazyLinear(num_cells, device=device),
                nn.Tanh(),
                nn.LazyLinear(num_cells, device=device),
                nn.Tanh(),
                nn.LazyLinear(num_cells, device=device),
                nn.Tanh(),
                nn.LazyLinear(n_outputs, device=device),
            )
        pass

    def forward(self, x):
        return self.net.forward(x)
