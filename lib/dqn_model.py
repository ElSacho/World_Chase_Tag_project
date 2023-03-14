import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_shape,hidden_size, n_actions):
        super(DQN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        x = self.net(x)
        F.softmax(x, dim=1)
        return F.softmax(x, dim=1)
        

    def forward2(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
