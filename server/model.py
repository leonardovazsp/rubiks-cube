import torch
from torch.nn import Module, Linear, ReLU, ModuleList, Conv2d, MaxPool2d, Flatten, Dropout, functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from data_pipeline import Dataset
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
    
class SingleImageModel(Module):
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
                 fc_sizes = [1024, 2048, 1024, 27 * 6],
                 dropout = 0.1):
        super().__init__()
        self.conv_blocks = ModuleList([ConvBlock(in_channels, out_channels, pool=pool) 
                                                for in_channels, out_channels, pool in zip(channels_list[:-1], channels_list[1:], pool_list)])
        output_shape = input_shape

        for pool, channel in zip(pool_list, channels_list[1:]):
            output_shape = (channel, output_shape[1] - kernel_size + 1, output_shape[2] - kernel_size + 1)
            if pool:
                output_shape = (output_shape[0], output_shape[1] // 2, output_shape[2] // 2)

        self.fc_sizes = [output_shape[0] * output_shape[1] * output_shape[2]] + fc_sizes

        self.fc = ModuleList([Linear(in_features, out_features) for in_features, out_features in zip(self.fc_sizes[:-1], self.fc_sizes[1:])])
        
        
    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = Flatten()(x)
        for i, fc in enumerate(self.fc):
            x = fc(x)
            if i < len(self.fc) - 1:
                x = ReLU()(x)
                x = Dropout(0.1)(x)

        return x.view(-1, 27, 6)
        
class Model(Module):
    """
    This model takes two images of a Rubik's Cube with a complete view of 3 faces
    and outputs the state of theses faces in the shape of (batch_size, 27, 6).

    Args:
        input_shape (tuple): shape of the input image
    """
    def __init__(self, input_shape=(3, 96, 96)):
        super().__init__()
        self.single_image_model = SingleImageModel(input_shape=input_shape)

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
        preds_1 = self.single_image_model(images_1)
        preds_2 = self.single_image_model(images_2)
        preds = self._process_outputs(preds_1, preds_2)
        return preds
    
    def training_step(self, batch):
        images, labels = batch
        labels = labels.view(-1, 54).long() # (batch_size, 6, 3, 3) -> (batch_size, 54)
        out = self(images) # (batch_size, 6, 54)
        loss = F.cross_entropy(out, labels)
        _, preds = torch.max(out, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / (len(preds)*54))
        return {'loss': loss, 'acc': acc}
    
    def validation_step(self, batch):
        images, labels = batch
        labels = labels.view(-1, 54).long() # (batch_size, 6, 3, 3) -> (batch_size, 54)
        out = self(images)
        loss = F.cross_entropy(out, labels)
        _, preds = torch.max(out, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / (len(preds)*54))
        return {'val_loss': loss.detach(), 'val_acc': acc}