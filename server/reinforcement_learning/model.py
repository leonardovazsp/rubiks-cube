import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class DQN(nn.Module):
    def __init__(self, 
                 input_shape = (6,3,3), 
                 n_actions = 12, 
                 lr=0.001, 
                 name = "model1.pt",
                 checkpoint_dir = "temp/") -> None:
        
        super(DQN, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        input_shape = np.prod(input_shape)

        self.fc1 = nn.Linear(input_shape, 512)
        self.fcmid = nn.Linear(512, 256) # additional layer
        self.fc2 = nn.Linear(256, n_actions)

        self._optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fcmid(x)) #additional forward
        x = self.fc2(x)
        return x

    def save_checkpoint(self) -> None:
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self) -> None:
        self.load_state_dict(torch.load(self.checkpoint_file))  

if __name__ == '__main__':
    dqn = DQN()

    state = torch.randint(0, 6, (6,3,3), dtype=torch.float).reshape(1, -1)
    action = torch.randint(0,12,(1,))
    state_ = torch.randint(0, 6, (6,3,3))
    reward = torch.multinomial(torch.tensor([3, 1], dtype=torch.float), 1)

    print(dqn(state))

    dqn.save_checkpoint()
    print(dqn.fc1.bias)

    # print(dqn.checkpoint_file)


