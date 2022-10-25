import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(6, 16)
        self.fc1 = nn.Linear(16 * 54, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, 6)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        dist = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return dist, value