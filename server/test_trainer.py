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
    'wandb_project': 'rubiks-cube-pose-estimation',
    'wandb_entity': 'leonardopereira',
    'save_dir': 'models'
}

if __name__ == '__main__':
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    dataset = Dataset(model_type='pose_estimation', resolution=(480, 480), directory='/mnt/data')
    model = models.PoseEstimator(input_shape = (3, 480, 480), pool_list = [True, True, True, True, False, False, True, True])
    
    trainer = Trainer(model=model, dataset=dataset, **train_params)
    profiler.enable()
    trainer.train()
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.TIME).print_stats(10)