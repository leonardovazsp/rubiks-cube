from model import ActorCritic
from environment import Cube
from collections import deque
from torch.optim.lr_scheduler import _LRScheduler
import torch
import numpy as np

class WarmupExponentialDecayLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, gamma, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        super(WarmupExponentialDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # linear warmup
            alpha = self.last_epoch / self.warmup_steps
            scale_factor = alpha

        else:
            # exponential decay
            exp_steps = self.last_epoch - self.warmup_steps
            scale_factor = self.gamma ** exp_steps

        return [base_lr * scale_factor for base_lr in self.base_lrs]

class Agent(ActorCritic):
    def __init__(self,
                 device='cuda',
                 batch_size=32,
                 lr=0.001,
                 warmup_steps=100,
                 gamma=0.9999,
                 checkpoint=None,
                 reward=1.0):
        
        super().__init__()
        self.device = device
        self.memory = deque(maxlen=batch_size)
        self.batch_size = batch_size
        self.policy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.value_loss = torch.nn.MSELoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = WarmupExponentialDecayLR(self.optimizer, warmup_steps=warmup_steps, gamma=gamma)
        self.checkpoint = checkpoint
        self.reward = reward
        self._init_model()

    def _init_model(self):
        if self.checkpoint:
            self.load_state_dict(torch.load(self.checkpoint))
        self.to(self.device)

    def expand_state(self, cube):
        children_states, rewards = [], []
        for move in range(12):
            next_state, reward = cube.get_next_state(move)
            children_states.append(next_state)
            rewards.append(reward)
        return children_states, rewards
    
    def generate_examples(self, scrambles):
        cube = Cube()
        cube.set_reward(self.reward)
        for i in range(self.batch_size // scrambles):
            cube.reset()
            for j in range(scrambles):
                cube.scramble(j)
                children, rewards = self.expand_state(cube)
                loss_discount = 1 / (j + 1)
                self.memory.append({'state': cube.state, 'children': children, 'rewards': rewards, 'loss_discount': loss_discount})

    def get_policy_value(self, children_states, rewards):
        self.eval()
        children_states = torch.tensor(np.array(children_states)).to(self.device).view(-1, 54)
        rewards = torch.tensor(np.array(rewards)).to(self.device).view(-1, 12)
        policies, values = self(children_states)
        values = values.view(-1, 12)
        values = values + rewards
        y_v, y_p = values.max(dim=1)
        return y_p.long(), y_v.float().unsqueeze(1)

    def train_step(self):
        states = [m['state'] for m in self.memory] # (batch_size, 54)
        children = [m['children'] for m in self.memory] # (batch_size, 12, 54)
        rewards = [m['rewards'] for m in self.memory] # (batch_size, 12)
        loss_discount = [m['loss_discount'] for m in self.memory]
        loss_discount = torch.tensor(loss_discount).float().to(self.device)
        y_p, y_v = self.get_policy_value(children, rewards)
        self.train()
        states = torch.tensor(np.array(states)).to(self.device)
        p, v = self(states)
        loss_p = self.policy_loss(p, y_p)
        loss_v = self.value_loss(v, y_v)
        loss = loss_p + loss_v
        loss *= loss_discount
        loss = loss.mean().float()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()
    
if __name__ == '__main__':
    agent = Agent(device='cuda', batch_size=8, lr=0.001, warmup_steps=100, gamma=0.9999)
    for i in range(10):
        agent.generate_examples(2)
        print(agent.train_step())
    print('Training complete!')