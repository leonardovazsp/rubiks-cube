import torch
from model import RubiksCubeModel
from data_pipeline import SampleGenerator, Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def get_model():
    model = RubiksCubeModel()
    to_device(model, device)
    return model

def get_data_loaders(train_ds, val_ds, batch_size):
    train_loader = DeviceDataLoader(DataLoader(train_ds, batch_size, shuffle=True, num_workers=12, pin_memory=True), device)
    val_loader = DeviceDataLoader(DataLoader(val_ds, batch_size, num_workers=12, pin_memory=True), device)
    return train_loader, val_loader

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD, gamma=0.999):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    scheduler = ExponentialLR(optimizer, gamma=0.999)
    for epoch in range(epochs):
        train_results = []
        # Training Phase 
        for batch in train_loader:
            result = model.training_step(batch)
            loss = result['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_results.append(result)
        scheduler.step()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack([x.get('loss') for x in train_results]).mean().item()
        result['train_acc'] = torch.stack([x.get('acc') for x in train_results]).mean().item()
        result['acc_1'] = torch.stack([x.get('acc_1') for x in train_results]).mean().item()
        result['acc_2'] = torch.stack([x.get('acc_2') for x in train_results]).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

    return history

if __name__ == '__main__':
    import numpy as np
    import json

    config = json.load(open('config.json'))
    batch_size = config['train']['batch_size']
    epochs = config['train']['epochs']
    gamma = config['train']['learning_rate_decay']
    lr = config['train']['learning_rate']
    lr = 0.05
    optimizer = torch.optim.SGD if 'optimizer' not in config or config['train']['optimizer'] == 'SGD' else torch.optim.Adam
    input_shape = config['train']['input_shape']
    resolution = input_shape[1:]
    
    device = get_default_device()
    data_dir = config['data_dir']

    train_ds = Dataset(resolution, data_dir, 'train', background_augmentation=0.5)
    val_ds = Dataset(resolution, data_dir, 'val', background_augmentation=0.5)
    train_loader, val_loader = get_data_loaders(train_ds, val_ds, batch_size)

    model = get_model()

    history = [evaluate(model, val_loader)]
    history += fit(epochs, lr, model, train_loader, val_loader, opt_func=optimizer, gamma=gamma)
    torch.save(model.state_dict(), 'models/model_val_acc_{:.4f}_train_acc_{:.4f}.pth'.format(history[-1]['val_acc'], history[-1]['train_acc']))
    