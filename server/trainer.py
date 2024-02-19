import os
import wandb
import torch
import torch.optim.lr_scheduler as lr_schedulers
from torch.utils.data import DataLoader, random_split
from dataset import Dataset
from tqdm import tqdm

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
        scheduler=None,
        lr_decay=None,
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
        self.scheduler = scheduler
        self.lr_decay = lr_decay
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
        if self.scheduler:
            self.scheduler = getattr(lr_schedulers, self.scheduler)(self.optimizer, gamma=self.lr_decay)
            
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
            running_loss = 0.0
            acc_history = []
            for i, batch in enumerate(tqdm(self.train_loader)):
                images, state = batch
                images = [x.to(self.device) for x in images]
                state = state.to(self.device)
                batch = [images, state]
                results = self.model.training_step(batch, self.optimizer, self.criterion)
                running_loss += results['loss'].item()
                acc_history.append(results['acc'])
                if self.scheduler:
                    self.scheduler.step()

                if self.wandb_project:
                    wandb.log({'loss': results['loss'].item(), 'accuracy': results['acc']})

            print(f'Epoch {epoch + 1} - loss: {running_loss / len(self.train_loader)} - accuracy: {sum(acc_history) / len(acc_history)}')

            running_loss = 0.0
            acc_history = []
            for i, batch in enumerate(tqdm(self.val_loader)):
                images, state = batch
                images = [x.to(self.device) for x in images]
                state = state.to(self.device)
                batch = [images, state]
                results = self.model.validation_step(batch, self.criterion)
                running_loss += results['val_loss'].item()
                acc_history.append(results['val_acc'])
                if self.wandb_project:
                    wandb.log({'val_loss': results['val_loss'].item(), 'val_accuracy': results['val_acc']})

            print(f'Epoch {epoch + 1} - val_loss: {running_loss / len(self.val_loader)} - val_accuracy: {sum(acc_history) / len(acc_history)}')
            if epoch % 10 == 0:
                    print("Learning rate:", self.optimizer.param_groups[0]['lr'])
                    
            acc = sum(acc_history) / len(acc_history)
            if save_model:
                if acc > save_criteria_score:
                    torch.save(self.model.state_dict(), f'{self.save_dir}/{self.model_name}_{self.run_name}_best.pth')
                    save_criteria_score = acc

        if save_model:
            torch.save(self.model.state_dict(), f'{self.save_dir}/{self.model_name}_{self.run_name}_latest.pth')