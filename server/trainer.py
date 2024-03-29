from torch.utils.data import DataLoader, random_split
from dataset import Dataset
import torch
from tqdm import tqdm
import os
import wandb

class Trainer():
    def __init__(self,
        model,
        dataset,
        optimizer='Adam',
        criterion='CrossEntropyLoss',
        device='cpu',
        batch_size=32,
        epochs=10,
        shuffle=True,
        split=0.2,
        lr=0.001,
        wandb_project=None,
        wandb_entity='leonardovaz',
        save_dir='models',
        *args,
        **kwargs
        ):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split
        self.lr = lr
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.save_dir = save_dir
        self.epochs = epochs
        self.kwargs = kwargs
        self._init_loaders()
        self._init_model()
        
    
    def _init_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def _init_loaders(self):
        print('Splitting dataset...')
        train_size = int((1 - self.split) * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_ds, val_ds = random_split(self.dataset, [train_size, val_size])
        print(f'Train size: {train_size}, Val size: {val_size}')

        print('Initializing loaders...')
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, num_workers=4, pin_memory=True)
        print('Loaders initialized!')

    def _init_model(self):
        print('Initializing model...')
        self.optimizer = getattr(torch.optim, self.optimizer)
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        self.criterion = getattr(torch.nn, self.criterion)()
        self.model.to(self.device)
        self.model.train()
        self.model_name = self.model.__class__.__name__
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print(f'Model {self.model_name} initialized!')

    def _init_wandb(self):
        if self.wandb_project:
            print('Initializing wandb...')
            model_config = self.model.config
            config = {
                'model': self.model.__class__.__name__,
                'optimizer': self.optimizer.__class__.__name__,
                'criterion': self.criterion.__class__.__name__,
                'device': self.device,
                'batch_size': self.batch_size,
                'shuffle': self.shuffle,
                'split': self.split,
                'lr': self.lr
            }
            config.update(model_config)
            wandb.init(project=self.wandb_project, entity=self.wandb_entity, config=config)
            wandb.watch(self.model)
            self.run_name = wandb.run.name
            print(f'wandb initialized! Run name: {self.run_name}')
        else:
            self.run_name = 'local'

    def train(self,
        epochs=None,
        save_model=True,
        ):
        if epochs is None:
            epochs = self.epochs
            
        self._init_wandb()
        save_criteria_score = 0
        print('Initializing training...')
        for epoch in range(epochs):
            # running_loss = 0.0
            # acc_history = []
            history = []
            for i, batch in enumerate(tqdm(self.train_loader)):
                images, state = batch
                images = [x.to(self.device) for x in images]
                state = state.to(self.device)
                batch = [images, state]
                results = self.model.training_step(batch, self.optimizer, self.criterion)
                history.append(results)
                # running_loss += results['loss'].item()
                # acc_history.append(results['acc'])
                if self.wandb_project:
                    wandb.log(results)
            print(f'Epoch {epoch + 1} - loss: {sum(h["loss"].item() for h in history) / len(history):.4f}' + \
                                   f' - accuracy: {sum(h["acc"] for h in history) / len(history):.4}' + \
                                   f' - precision: {sum(h["precision"] for h in history) / len(history):.4f}' + \
                                   f' - recall: {sum(h["recall"] for h in history) / len(history):.4f}' + \
                                   f' - f1: {sum(h["f1"] for h in history) / len(history):.4f}')

            # running_loss = 0.0
            # acc_history = []
            history = []
            for i, batch in enumerate(tqdm(self.val_loader)):
                images, state = batch
                images = [x.to(self.device) for x in images]
                state = state.to(self.device)
                batch = [images, state]
                results = self.model.validation_step(batch, self.criterion)
                history.append(results)
                # running_loss += results['val_loss'].item()
                # acc_history.append(results['val_acc'])
                if self.wandb_project:
                    wandb.log(results)

            print(f'Epoch {epoch + 1} - val_loss: {sum(h["val_loss"].item() for h in history) / len(history):.4f}' + \
                                   f' - val_accuracy: {sum(h["val_acc"] for h in history) / len(history):.4}' + \
                                   f' - val_precision: {sum(h["val_precision"] for h in history) / len(history):.4f}' + \
                                   f' - val_recall: {sum(h["val_recall"] for h in history) / len(history):.4f}' + \
                                   f' - val_f1: {sum(h["val_f1"] for h in history) / len(history):.4f}')

            f1 = sum(h["val_f1"] for h in history) / len(history)
            if save_model:
                if f1 > save_criteria_score:
                    torch.save(self.model.state_dict(), f'{self.save_dir}/{self.model_name}_{self.run_name}_best.pth')
                    save_criteria_score = f1
                    if self.wandb_project:
                        wandb.run.summary['best_f1'] = f1

        if save_model:
            torch.save(self.model.state_dict(), f'{self.save_dir}/{self.model_name}_{self.run_name}_latest.pth')