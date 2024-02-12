from dataset import Dataset
from trainer import Trainer
import model as models
import torch
import json

train_params = {
    'epochs': 10,
    'batch_size': 4,
    'lr': 0.001,
    'shuffle': True,
    'save_model': True,
    'save_history': True,
    'split': 0.2,
    'optimizer': 'Adam',
    'criterion': 'CrossEntropyLoss',
    'device': 'cuda',
    'wandb_project': 'rubiks-cube-color-recognition',
    'save_dir': 'models'
}

if __name__ == '__main__':
    dataset = Dataset()
    model = models.ColorRecognizer()
    trainer = Trainer(model=model, dataset=dataset, **train_params)
    trainer.train()
