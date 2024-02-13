from dataset import Dataset
from trainer import Trainer
import model as models
import torch
import json

train_params = {
    'epochs': 100,
    'batch_size': 4,
    'lr': 0.0001,
    'shuffle': True,
    'save_model': True,
    'save_history': True,
    'split': 0.1,
    'optimizer': 'Adam',
    'criterion': 'MSELoss',
    'device': 'cuda',
    'wandb_project': 'rubiks-cube-pose-estimation',
    'save_dir': 'models'
}

if __name__ == '__main__':
    dataset = Dataset(model_type='pose_estimation')
    model = models.PoseEstimator()
    trainer = Trainer(model=model, dataset=dataset, **train_params)
    trainer.train()
