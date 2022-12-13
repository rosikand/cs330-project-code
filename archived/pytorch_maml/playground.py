import torchplate
from torchplate import experiment
from torchplate import utils
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import requests
import pdb
import cloudpickle as cp
from urllib.request import urlopen
import rsbox 
from rsbox import ml, misc
import pdb
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1*28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 50)

    def forward(self, x):
        # grayscale to rgb if needed 
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# yo = Net()
# for param in yo.parameters():
#     pdb.set_trace()


accuracies = [
    [1, 0.2, 0.3, 0.4, 0.5],
    [3, 0.3, 0.4, 0.5, 0.6],
    [4, 0.4, 0.5, 0.6, 0.7],
]

print(np.mean(accuracies, axis=0))
