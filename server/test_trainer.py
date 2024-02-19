from dataset import Dataset
from trainer import Trainer
import model as models
import torch
import json

train_params = {
    'epochs': 200,
    'batch_size': 32,
    'lr': 0.00005,
    'shuffle': True,
    'save_model': True,
    'save_history': True,
    'split': 0.1,
    'optimizer': 'Adam',
    'scheduler': 'ExponentialLR',
    'lr_decay': 0.9995,
    'criterion': 'CrossEntropyLoss',
    'device': 'cuda',
    'wandb_project': 'rubiks-cube-color-recognition',
    'save_dir': 'models'
}

if __name__ == '__main__':
    resolution = (240, 240)
    dataset = Dataset(model_type='color_recognition', resolution=resolution)
    model = models.ColorRecognizer(input_shape=(3, *resolution))
    trainer = Trainer(model=model, dataset=dataset, **train_params)
    trainer.train()
