import torch.nn as nn
import torch.nn.functional as F

# ==========
# Base Model
# ==========

class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()

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
            data += str(l.in_features) + 'x'
        data += str(l.out_features)
        return data

    def get_name(self):
        """
        Return the name of this model.
        """
        return 'dqn' + self.extract_model_data()
    
    def get_filename(self):
        """
        Returns the file name of the given model.
        """
        return 'weights' + '/' + self.get_name() + ".pth"


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
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        pass
         
    def forward(self, x):
        """
        Determine the next action from one element or a batch during optimization.

        Returns `tensor([[left0exp,right0exp]...])`.
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    pass


# =========
# PPO Model
# =========

# TODO: define and implement