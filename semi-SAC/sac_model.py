import torch.nn as nn
import torch.nn.functional as F
import torch


class SACActor(nn.Module):
    def __init__(self, num_layer, dimS, nA, hidden1, hidden2, hidden3):
        super(SACActor, self).__init__()
        self.num_layer = num_layer
        self.nA = nA
        self.fc1 = nn.Linear(dimS, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        if self.num_layer == 2:
            self.fc3 = nn.Linear(hidden2, nA)
        else:
            self.fc3 = nn.Linear(hidden2, hidden3)
            self.fc4 = nn.Linear(hidden3, nA)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        if self.num_layer == 2:
            p = F.softmax(self.fc3(x), dim=-1)
        else:
            x = F.relu(self.fc3(x))
            p = F.softmax(self.fc4(x), dim=-1)
        return p