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
    train_loader = DeviceDataLoader(DataLoader(train_ds, batch_size, shuffle=True, num_workers=28, pin_memory=True), device)
    val_loader = DeviceDataLoader(DataLoader(val_ds, batch_size, num_workers=28, pin_memory=True), device)
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
        model.epoch_end(epoch, result)
        history.append(result)
        # Early stopping
        patience = 5
        if len(history) > patience and result['val_acc'] < min([x['val_acc'] for x in history[-patience-1:-1]]):
            print("Early stopping...")
            break
    return history

if __name__ == '__main__':
    import wandb
    import numpy as np
    import json
    
    config = json.load(open('config.json'))

    device = get_default_device()
    data_dir = config['data_dir']

    dropouts = np.linspace(0, 0.25, 10)
    batch_sizes = [8, 16, 32, 64]
    epochs = 10
    gammas = np.linspace(0.9, 0.9999, 10)
    lrts = np.logspace(-2, -0.4, 50)
    optimizers = [torch.optim.Adam, torch.optim.SGD]
    resolution = (96, 96)

    for i in range(100):
        dropout = np.random.choice(dropouts).item()
        batch_size = np.random.choice(batch_sizes).item()
        lr = np.random.choice(lrts).item()
        gamma = np.random.choice(gammas).item()
        optimizer = np.random.choice(optimizers)
        train_ds = Dataset(resolution, data_dir, 'train')
        val_ds = Dataset(resolution, data_dir, 'val')
        train_size = len(train_ds)
        train_loader, val_loader = get_data_loaders(train_ds, val_ds, batch_size)
        model = get_model()
        model.dropout1.p = dropout
        model.dropout2.p = dropout
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())

        wandb.init(project='rubiks-cube',
                    entity="leonardovaz",
                    config={'batch_size': batch_size, 
                            'dropout': dropout,
                            'epochs': epochs, 
                            'learning_rate': lr,
                            'learning_decay': gamma,
                            "train_size": train_size,
                            "img_size": resolution,
                            "model_params": sum([np.prod(p.size()) for p in model_parameters]),
                            "conv_layers": {'filters':[8, 16, 32, 64, 128, 256, 512, 1024],
                                            'kernel_size': [3, 3, 3, 3, 3, 3, 3, 3]},
                            "linear_layers": [1024 * 2 * 2, 1024, 2048, 1024],
                            "layernorm": True,
                            "batchnorm": True,
                            "activation": 'ReLU',
                            "optimizer": optimizer.__name__,
                            "background_augmentation": True})

        wandb.watch(model)

        history = [evaluate(model, val_loader)]
        history += fit(epochs, lr, model, train_loader, val_loader, opt_func=optimizer, gamma=gamma)
        torch.save(model.state_dict(), 'models/model_val_acc_{:.4f}_train_acc_{:.4f}.pth'.format(history[-1]['val_acc'], history[-1]['train_acc']))
        wandb.finish()