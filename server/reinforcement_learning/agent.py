import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .model import Model

class ReplayBuffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        batch = np.random.choice(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class Agent():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model(args).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.replay_buffer = ReplayBuffer(args)
        self.writer = SummaryWriter()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        dist, _ = self.model(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample()
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        dist, value = self.model(state)
        _, next_value = self.model(next_state)
        advantage = reward + self.args.gamma * next_value * (1 - done) - value

        actor_loss = -(dist.log_prob(action) * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar('loss/actor_loss', actor_loss.item(), self.replay_buffer.ptr)
        self.writer.add_scalar('loss/critic_loss', critic_loss.item(), self.replay_buffer.ptr)
        self.writer.add_scalar('loss/loss', loss.item(), self.replay_buffer.ptr)
        self.writer.add_scalar('values/value', value.mean().item(), self.replay_buffer.ptr)
        self.writer.add_scalar('values/next_value', next_value.mean().item(), self.replay_buffer.ptr)
        self.writer.add_scalar('values/reward', reward.mean().item(), self.replay_buffer.ptr)

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))