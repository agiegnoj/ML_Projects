import torch
import torch.nn as nn

class Network(nn.Module):
    """MLP for PPO"""
    def __init__(self,  numObservations, numActions):
        super().__init__()
        self.layers = nn.Sequential(
           nn.Linear(numObservations, 256),
           nn.Tanh(),
           nn.Dropout(0.1),
           nn.Linear(256, 256),
           nn.Tanh(),
           nn.Dropout(0.1),
           nn.Linear(256, 256),
           nn.Tanh(),
           nn.Dropout(0.1),
           nn.Linear(256, numActions),
       )
    def forward(self, x):
        return torch.tanh(self.layers(x))
