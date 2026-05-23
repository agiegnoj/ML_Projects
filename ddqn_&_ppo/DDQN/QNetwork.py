import torch.nn as nn

class QNetwork(nn.Module):
    """MLP for DQN"""
    def __init__(self, observationDim, actionDim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observationDim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, actionDim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.layers(x)
