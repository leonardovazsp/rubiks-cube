from model import ActorCritic
from environment import Cube
from collections import deque
from torch.optim.lr_scheduler import _LRScheduler
import torch
import numpy as np
import time

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

class Model(ActorCritic):
    def __init__(self,
                 device='cuda',
                 batch_size=32,
                 lr=0.001,
                 optimizer='Adam',
                 weight_decay=0.0001,
                 warmup_steps=100,
                 gamma=0.9999,
                 checkpoint=None,
                 shuffle=True):
        
        super().__init__()
        self.device = device
        self.memory = None
        self.batch_size = batch_size
        self.policy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.value_loss = torch.nn.MSELoss(reduction='none')
        self.optimizer = getattr(torch.optim, optimizer)(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = WarmupExponentialDecayLR(self.optimizer, warmup_steps=warmup_steps, gamma=gamma)
        self.checkpoint = checkpoint
        self.shuffle = shuffle
        self._init_model()

    def _init_model(self):
        if self.checkpoint:
            self.load_state_dict(torch.load(f'models/{self.checkpoint}', map_location=self.device))
            
        self.to(self.device)

    def set_memory(self, queue):
        self.memory = queue

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
        memory = self.memory.get()
        states = [m['state'] for m in memory] # (batch_size, 54)
        children = [m['children'] for m in memory] # (batch_size, 12, 54)
        rewards = [m['rewards'] for m in memory] # (batch_size, 12)
        loss_discount = [m['loss_discount'] for m in memory]
        solved = [m['solved'] for m in memory]
        loss_discount = torch.tensor(loss_discount).float().to(self.device)
        solved = torch.tensor(solved).float().to(self.device)
        y_p, y_v = self.get_policy_value(children, rewards)
        # y_v += solved.unsqueeze(-1) * self.reward
        # print(y_v.view(-1, self.scrambles))
        # for i in range(5):
        #     print(f"Value: {y_v[i].item()}:")
        #     print(Cube(np.array(states[i])))
        # time.sleep(5)
        self.train()
        states = torch.tensor(np.array(states)).to(self.device)
        p, v = self(states)
        loss_p = self.policy_loss(p, y_p)# * (1 - solved)
        loss_v = self.value_loss(v, y_v)
        loss = loss_p + loss_v
        loss *= loss_discount
        loss = loss.mean().float()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

class Agent():
    def __init__(self,
                 memory,
                 batch_size,
                 reward,
                 scrambles):
        self.memory = memory
        self.batch_size = batch_size
        self.reward = reward
        self.scrambles = scrambles

    def expand_state(self, cube):
        children_states, rewards = [], []
        for move in range(12):
            next_state, reward = cube.get_next_state(move)
            children_states.append(next_state)
            rewards.append(reward)
        return children_states, rewards

    def generate_examples(self, stop_signal):
        count = 0
        while not stop_signal.value:
            if self.memory.qsize() >= self.batch_size * 16:
                time.sleep(1)
                continue

            batch = []
            for i in range(self.batch_size // self.scrambles):
                cube = Cube()
                cube.set_reward(self.reward)
                for j in range(1, self.scrambles):
                    cube.scramble(j)
                    children, rewards = self.expand_state(cube)
                    loss_discount = 1 / (j + 1)
                    solved = j == 0
                    batch.append({'state': cube.state, 'children': children, 'rewards': rewards, 'loss_discount': loss_discount, 'solved': solved})
                    count += 1

            self.memory.put(batch)

    
    
if __name__ == '__main__':
    agent = Agent(device='cuda', batch_size=8, lr=0.001, warmup_steps=100, gamma=0.9999)
    for i in range(10):
        agent.generate_examples(2)
        print(agent.train_step())
    print('Training complete!')