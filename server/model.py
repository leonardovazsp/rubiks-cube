import torch
from torch.nn import Module, Linear, ReLU, ModuleList, Conv2d, MaxPool2d, Flatten, Dropout, functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
# from data_pipeline import Dataset
from torch.optim.lr_scheduler import ExponentialLR

class ConvBlock(Module):
    """
    Building block for the convolutional layers.
    It consists of a convolutional layer, a batch normalization layer, and a max
    pooling layer, which may or may not be included.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        pool (bool): whether to include a max pooling layer
        dropout (float): dropout rate

    """
    def __init__(self, in_channels, out_channels, pool=False, dropout=None):
        super().__init__()
        self.dropout = Dropout(dropout) if dropout else None
        self.conv = Conv2d(in_channels, out_channels, 3)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.pool = MaxPool2d(2, 2) if pool else None

    def forward(self, x):
        x = ReLU()(self.conv(x))
        if self.dropout:
            x = self.dropout(x)

        x = self.batch_norm(x)
        return self.pool(x) if self.pool else x
    
class Encoder(Module):
    """
    This model takes an image of a Rubik's Cube with a complete view of 3 faces
    and outputs the state of theses faces in the shape of (batch_size, 27, 6).

    Args:
        input_shape (tuple): shape of the input image
        kernel_size (int): size of the convolutional kernel
        channels_list (list): list of the number of channels for each convolutional layer
        pool_list (list): list of booleans indicating whether to include a max pooling layer
        fc_sizes (list): list of the number of neurons for each fully connected layer
        dropout (float): dropout rate
    """
    def __init__(self,
                 input_shape = (3, 96, 96),
                 kernel_size = 3,
                 channels_list = [3, 8, 16, 32, 64, 128, 256, 512, 1024],
                 pool_list = [True, True, False, False, False, False, True, True],
                 fc_sizes = [1024, 2048, 1024],
                 dropout = 0.1):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.channels_list = channels_list
        self.pool_list = pool_list
        self.fc_sizes = fc_sizes
        self.dropout = dropout

        self.conv_blocks = ModuleList([ConvBlock(in_channels, out_channels, pool=pool) 
                                                for in_channels, out_channels, pool in zip(channels_list[:-1], channels_list[1:], pool_list)])
        output_shape = input_shape

        for pool, channel in zip(pool_list, channels_list[1:]):
            output_shape = (channel, output_shape[1] - kernel_size + 1, output_shape[2] - kernel_size + 1)
            if pool:
                output_shape = (output_shape[0], output_shape[1] // 2, output_shape[2] // 2)

        self.fc_sizes = [output_shape[0] * output_shape[1] * output_shape[2]] + fc_sizes

        self.fc = ModuleList([Linear(in_features, out_features) for in_features, out_features in zip(self.fc_sizes[:-1], self.fc_sizes[1:])])
        
    @property
    def config(self):
        return {
            'input_shape': self.input_shape,
            'kernel_size': self.kernel_size,
            'channels_list': self.channels_list,
            'pool_list': self.pool_list,
            'fc_sizes': self.fc_sizes,
            'dropout': self.dropout
        }

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = Flatten()(x)
        for i, fc in enumerate(self.fc):
            x = fc(x)
            if i < len(self.fc) - 1:
                x = ReLU()(x)
                x = Dropout(0.1)(x)

        return x
        
class ColorRecognizer(Module):
    """
    This model takes two images of a Rubik's Cube with a complete view of 3 faces
    and outputs the state of theses faces in the shape of (batch_size, 27, 6).

    Args:
        input_shape (tuple): shape of the input image
    """
    def __init__(self,
                 input_shape = (3, 96, 96),
                 kernel_size = 3,
                 channels_list = [3, 8, 16, 32, 64, 128, 256, 512, 1024],
                 pool_list = [True, True, False, False, False, False, True, True],
                 fc_sizes = [1024, 2048, 1024],
                 dropout = 0.1):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.channels_list = channels_list
        self.pool_list = pool_list
        self.fc_sizes = fc_sizes
        self.dropout = dropout

        self.encoder = Encoder(input_shape=input_shape,
                                kernel_size=kernel_size,
                                channels_list=channels_list,
                                pool_list=pool_list,
                                fc_sizes=fc_sizes,
                                dropout=dropout
                                )
        self.linear = Linear(fc_sizes[-1] * 2, 54 * 6)

    @property
    def config(self):
        return {
            'input_shape': self.input_shape,
            'kernel_size': self.kernel_size,
            'channels_list': self.channels_list,
            'pool_list': self.pool_list,
            'fc_sizes': self.fc_sizes,
            'dropout': self.dropout
        }

    def _process_outputs(self, out1, out2):
        # shape of out1 and out2: (batch_size, 27, 6)

        # Stack the two outputs from each image
        outputs = torch.stack([out1, out2], dim=1) # (batch_size, 54, 6)

        # Reshape the outputs and transpose so that the value for each shape is in dimension 1
        outputs = outputs.view(outputs.shape[0], 6, 54).permute(1, 0, 2) # (6, batch_size, 54)

        # Adjust the order of the faces and permute the dimensions so that the batch size is the first dimension again
        # Image 1 sees faces (0, 1, 4) and Image 2 sees faces (2, 3, 5)
        # The output stack was (0, 1, 4, 2, 3, 5) and we want (0, 1, 2, 3, 4, 5)
        outputs = outputs[[0, 1, 3, 4, 2, 5]].permute(1, 2, 0) # (batch_size, 54, 6)

        # Reshape the outputs so that the last dimension is the 6 values for each face
        outputs = outputs.reshape(-1, 6, 54) # (batch_size, 6, 54)
        return outputs
    
    def forward(self, batch):
        images_1, images_2 = batch
        encoded_1 = self.encoder(images_1)
        encoded_2 = self.encoder(images_2)
        preds = self.linear(torch.cat([encoded_1, encoded_2], dim=1))
        return preds.view(-1, 6, 54)
    
    def training_step(self, batch, optimizer, criterion):
        self.train()
        images, labels = batch
        labels = labels.view(-1, 54).long() # (batch_size, 6, 3, 3) -> (batch_size, 54)
        out = self(images) # (batch_size, 6, 54)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, preds = torch.max(out, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / (len(preds)*54))
        return {'loss': loss.detach(), 'acc': acc}
    
    def validation_step(self, batch, criterion):
        self.eval()
        images, labels = batch
        labels = labels.view(-1, 54).long() # (batch_size, 6, 3, 3) -> (batch_size, 54)
        out = self(images)
        loss = criterion(out, labels)
        _, preds = torch.max(out, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / (len(preds)*54))
        return {'val_loss': loss.detach(), 'val_acc': acc}

class PoseEstimator(Module):
    def __init__(self,
                 input_shape = (3, 96, 96),
                 kernel_size = 3,
                 channels_list = [3, 8, 16, 32, 64, 128, 256, 512, 1024],
                 pool_list = [True, True, False, False, False, False, True, True],
                 fc_sizes = [1024, 2048, 1024],
                 dropout = 0.1):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.channels_list = channels_list
        self.pool_list = pool_list
        self.fc_sizes = fc_sizes
        self.dropout = dropout

        self.encoder = Encoder(input_shape=input_shape,
                                kernel_size=kernel_size,
                                channels_list=channels_list,
                                pool_list=pool_list,
                                fc_sizes=fc_sizes,
                                dropout=dropout
                                )
        
        self.linear = Linear(fc_sizes[-1], 6)

    @property
    def config(self):
        return {
            'input_shape': self.input_shape,
            'kernel_size': self.kernel_size,
            'channels_list': self.channels_list,
            'pool_list': self.pool_list,
            'fc_sizes': self.fc_sizes,
            'dropout': self.dropout
        }
    
    def forward(self, image):
        encoded = self.encoder(image)
        preds = self.linear(encoded)
        return preds.view(-1, 6)
    
    def training_step(self, batch, optimizer, criterion):
        self.train()
        images, labels = batch
        image = images[0]
        labels = labels.view(-1, 6).float() # (batch_size, 6, 3, 3) -> (batch_size, 54)
        out = self(image) # (batch_size, 6, 54)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        preds = torch.round(out)
        acc = torch.tensor(torch.sum(preds == labels).item() / (len(preds)*6))
        return {'loss': loss.detach(), 'acc': acc}
    
    def validation_step(self, batch, criterion):
        self.eval()
        images, labels = batch
        image = images[0]
        labels = labels.view(-1, 6).float()
        out = self(image)
        loss = criterion(out, labels)
        preds = torch.round(out)
        acc = torch.tensor(torch.sum(preds == labels).item() / (len(preds)*6))
        return {'val_loss': loss.detach(), 'val_acc': acc}