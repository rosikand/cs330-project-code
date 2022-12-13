"""
File: modules.py
------------------
Nets. 
"""

# PyTorch nn.modules 
import torch 
import torch.nn as nn
import torch.nn.functional as F



def get_network(num_classes):
    # returns the CNN with output size num_classes 
    class NWayCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(3*128*128, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 32)
            self.fc4 = nn.Linear(32, num_classes)

        def forward(self, x):
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x
    
    net = NWayCNN()
    return net

