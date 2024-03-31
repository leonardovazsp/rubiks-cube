import torch
import torch.nn as nn
import numpy as np

def scaled_dot_product(q, k, v, mask=None):
    dk = q.shape[-1]
    qk = torch.matmul(q, k.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(dk))
    if mask:
        dk = dk + mask * float('-inf')

    qk = torch.softmax(qk, dim=-1)
    att = torch.matmul(qk, v)
    return att

def positional_encoding(max_seq_len, emb_dim):
    pe = torch.zeros(max_seq_len, emb_dim)
    for pos in range(max_seq_len):
        for i in range(0, emb_dim, 2):
            pe[pos, i] = np.sin(pos/(10000**((2*i)/emb_dim)))
            if i + 1 < emb_dim:
                pe[pos, i + 1] = np.cos(pos/(10000**((2*i)/emb_dim)))
    return torch.tensor(pe)

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.proj_dim = d_model // num_heads
        self.num_heads = num_heads
        self.q_projection = nn.Linear(d_model, d_model, bias=False)
        self.k_projection = nn.Linear(d_model, d_model, bias=False)
        self.v_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, d_model = q.shape

        q = self.q_projection(q)
        k = self.k_projection(k)
        v = self.v_projection(v)

        q = q.view(batch_size, seq_len, self.num_heads, self.proj_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.proj_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.proj_dim).permute(0, 2, 1, 3)

        att = scaled_dot_product(q, k, v, mask)
        att = att.view(batch_size, seq_len, d_model)
        return att
    
class FeedForward(nn.Module):
    def __init__(self, d_model, dff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dff)
        self.fc2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
class EncoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, dff, dropout=0.1):
        super().__init__()
        self.multihead_attention = MultiheadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dff)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        att = self.multihead_attention(x, x, x, mask)
        out_1 = self.layer_norm_1(x + self.dropout(att))
        ff = self.feed_forward(out_1)
        out_2 = self.layer_norm_2(out_1 + self.dropout(ff))
        return out_2
    
class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, dff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(num_heads, d_model, dff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)

        return x

class Patchify(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = self.unfold(x)

        return x.view(batch_size, channels, self.patch_size, self.patch_size, -1).permute(0, 4, 1, 2, 3)

class VisionEncoder(nn.Module):
    def __init__(self, max_len, patch_dim, num_layers, num_heads, d_model, dff, dropout):
        super().__init__()
        self.patch_dim = patch_dim
        self.patchify = Patchify(patch_dim)
        self.pos_encoding = positional_encoding(max_len, d_model)
        self.projection = nn.Linear(patch_dim**2 * 3, d_model)
        self.encoder = Encoder(num_layers, num_heads, d_model, dff, dropout)
        self.cls = nn.Embedding(1, d_model)
        self.batch_norm = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.batch_norm(x)
        batch_size, c, h, w = x.shape
        patches = self.patchify(x)
        batch_size, seq_len, c, h, w = patches.shape
        flattened_patches = patches.view(batch_size, seq_len, -1)
        projections = self.projection(flattened_patches) # (batch_size, seq_len, d_model)
        cls = self.cls.weight.repeat(batch_size, 1, 1)
        projections = torch.cat([cls, projections], dim=1)
        inputs = projections + self.pos_encoding[None, :seq_len + 1, :].to(projections.device)
        out = self.encoder(inputs)
        return out[:, 0, :]
    
class ColorRecognizer(nn.Module):
    def __init__(self, max_len, patch_dim, num_layers, num_heads, d_model, dff, dropout):
        super().__init__()
        self.vision_encoder = VisionEncoder(max_len, patch_dim, num_layers, num_heads, d_model, dff, dropout)
        self.linear = nn.Linear(d_model * 2, 54 * 6)

    def forward(self, batch):
        images_1, images_2 = batch
        encoded_1 = self.vision_encoder(images_1)
        encoded_2 = self.vision_encoder(images_2)
        preds = self.linear(torch.cat([encoded_1, encoded_2], dim=1))
        return preds.view(-1, 6, 3, 3, 6)
    
    def _get_one_hot_from_pred(self, preds):
        preds = torch.argmax(preds, dim=-1)
        identity = torch.eye(6).to(preds.device)
        one_hot_state = identity[preds.long()]
        return one_hot_state
    
    def training_step(self, batch, optimizer, criterion):
        self.train()
        images, labels = batch
        identity = torch.eye(6).to(labels.device)
        one_hot_labels = identity[labels.long()].permute(0, 4, 1, 2, 3).float() # (batch_size, 6, 3, 3) -> (batch_size, 6, 3, 3, 6)
        out = self(images) # (batch_size, 6, 3, 3, 6)
        out = out.permute(0, 4, 1, 2, 3) # (batch_size, 6, 3, 3, 6) -> (batch_size, 6, 6, 3, 3)
        loss = criterion(out, one_hot_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, preds = torch.max(out, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / (len(preds)*54))
        return {'loss': loss.detach(), 'acc': acc}

    def validation_step(self, batch, criterion):
        self.eval()
        images, labels = batch
        identity = torch.eye(6).to(labels.device)
        one_hot_labels = identity[labels.long()].permute(0, 4, 1, 2, 3).float()
        out = self(images)
        out = out.permute(0, 4, 1, 2, 3)
        loss = criterion(out, one_hot_labels)
        _, preds = torch.max(out, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / (len(preds)*54))
        return {'val_loss': loss.detach(), 'val_acc': acc}