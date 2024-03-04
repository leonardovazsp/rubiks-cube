import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.embedding = nn.Embedding(6, 256)
        self.fc1 = nn.Linear(256 * 54, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(2048, 1024)
        self.policy = nn.Linear(1024, 12)
        self.value = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out1 = F.relu(self.fc3(x))
        out2 = F.relu(self.fc4(x))
        p = self.policy(out1)
        p = F.softmax(p, dim=-1)
        v = self.value(out2)
        return p, v

    def compile(self, optimizer, losses):
        self.optimizer = optimizer
        self.losses = losses

    def train_step(self, batch):
        p, v = self(batch['state'])
        loss_p = self.losses['ce'](p, batch['y_p'])
        loss_v = self.losses['mse'](v, batch['y_v'])
        loss = loss_p + loss_v
        loss *= batch['loss_discount']
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[None, :x.size(1), :]
        return x

class Transformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dff=512, num_layers=8, dropout=0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model)
        self.embedding = nn.Embedding(6, d_model)
        self.encoder = nn.TransformerEncoderLayer(d_model, num_heads, dff, dropout)
        self.policy = nn.Linear(d_model, 12)
        self.value = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = torch.cat([self.cls.repeat(x.size(0), 1, 1), x], dim=1)
        x = self.pos_encoder(x)
        x = self.encoder(x).mean(dim=1)
        p = self.policy(x)
        v = self.value(x)
        return p, v

    def compile(self, optimizer, losses):
        self.optimizer = optimizer
        self.losses = losses

    def train_step(self, batch):
        p, v = self(batch['state'])
        loss_p = self.losses['ce'](p, batch['y_p'])
        loss_v = self.losses['mse'](v, batch['y_v'])
        loss = loss_p + loss_v
        loss *= batch['loss_discount']
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()