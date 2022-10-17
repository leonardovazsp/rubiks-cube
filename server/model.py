import torch
from torch.nn import Module, Linear, ReLU, Sequential, Conv2d, MaxPool2d, Flatten, Dropout, BatchNorm2d, LayerNorm, BatchNorm1d, functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from data_pipeline import Dataset
from torch.optim.lr_scheduler import ExponentialLR

class ConvBlock(Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super().__init__()
        self.conv = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, stride=1),
            ReLU(),
            BatchNorm2d(out_channels)
        )

        self.pool = MaxPool2d(2, 2) if pool else None

    def forward(self, x):
        return self.pool(self.conv(x)) if self.pool else self.conv(x)

class SingleImageModel(Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = ConvBlock(3, 8, pool=True)
        self.conv_block2 = ConvBlock(8, 16, pool=True)
        self.conv_block3 = ConvBlock(16, 32, pool=False)
        self.conv_block4 = ConvBlock(32, 64, pool=False)
        self.conv_block5 = ConvBlock(64, 128, pool=False)
        self.conv_block6 = ConvBlock(128, 256, pool=False)
        self.conv_block7 = ConvBlock(256, 512, pool=True)
        self.conv_block8 = ConvBlock(512, 1024, pool=True)
        self.fc1 = Linear(1024 * 2 * 2, 1024)
        self.fc2 = Linear(1024, 2048)
        self.fc3 = Linear(2048, 1024)
        self.fc4 = Linear(1024, 27*6)
        self.relu = ReLU()
        self.dropout = Dropout(0.02)

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        out = self.conv_block4(out)
        out = self.conv_block5(out)
        out = self.conv_block6(out)
        out = self.conv_block7(out)
        out = self.conv_block8(out)
        out = out.view(-1, 1024 * 2 * 2)
        out = self.relu(self.dropout(self.fc1(out)))
        out = self.relu(self.dropout(self.fc2(out)))
        out = self.relu(self.dropout(self.fc3(out)))
        out = self.fc4(out)
        return out.view(-1, 27, 6)

    def training_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)
        _, preds = torch.max(out, dim=-1)
        _, labels = torch.max(labels, dim=-1)
        acc = torch.tensor(torch.sum(preds==labels).item() / (len(preds)*27))
        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)  
        loss = F.cross_entropy(out, labels)
        _, preds = torch.max(out, dim=-1)
        _, labels = torch.max(labels, dim=-1)
        acc = torch.tensor(torch.sum(preds == labels).item() / (len(preds)*27))
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f} val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc']))

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def predict(self, images):
        outputs = self(images)
        _, preds = torch.max(torch.sum(outputs, dim=0), dim=-1)
        return preds

class RubiksCubeModel(Module):
    def __init__(self):
        super(RubiksCubeModel, self).__init__()
        self.img = SingleImageModel()
        # self.img_2 = SingleImageModel()
        self.fc1 = Linear(54 * 6, 64)
        self.fc2 = Linear(64, 128)
        self.fc3 = Linear(128, 256)
        self.fc4 = Linear(256, 128)
        self.fc5 = Linear(128, 54 * 6)
        self.batch_norm1 = BatchNorm1d(64)
        self.batch_norm2 = BatchNorm1d(128)
        self.batch_norm3 = BatchNorm1d(256)
        self.batch_norm4 = BatchNorm1d(128)
        self.relu = ReLU()
        self.dropout = Dropout(0)

    def forward(self, x):
        x0 = x[0]
        # print(x0)
        x1 = x[1]
        out1 = self.img(x0)
        out2 = self.img(x1)
        out = torch.cat((out1.view(-1, 27*6), out2.view(-1, 27*6)), dim=-1)
        out = self.relu(self.batch_norm1(self.fc1(out)))
        out = self.relu(self.batch_norm2(self.fc2(out)))
        out = self.relu(self.batch_norm3(self.fc3(out)))
        out = self.relu(self.batch_norm4(self.fc4(out)))
        out = self.fc5(out)
        out = out.view(-1, 54, 6)
        return out1, out2, out

    def training_step(self, batch):
        images, labels = batch 
        out1, out2, out = self(images)
        loss_1 = F.cross_entropy(out1, labels[0])
        loss_2 = F.cross_entropy(out2, labels[1])
        loss_3 = F.cross_entropy(out, labels[2])
        loss = torch.mean(torch.stack((loss_1, loss_2, loss_3)))
        # loss = loss_1 * self.acc_gap[0] + loss_2 * self.acc_gap[1] + loss_3 * self.acc_gap[2]
        _, preds_1 = torch.max(out1, dim=-1)
        _, labels_1 = torch.max(labels[0], dim=-1)
        _, preds_2 = torch.max(out2, dim=-1)
        _, labels_2 = torch.max(labels[1], dim=-1)
        _, preds = torch.max(out, dim=-1)
        _, labels_ = torch.max(labels[2], dim=-1)
        acc_1 = torch.tensor(torch.sum(preds_1==labels_1).item() / (len(preds_1)*27))
        acc_2 = torch.tensor(torch.sum(preds_2==labels_2).item() / (len(preds_1)*27))
        acc = torch.tensor(torch.sum(preds==labels_).item() / (len(preds)*54))
        return {'loss': loss, 'acc': acc, 'acc_1': acc_1, 'acc_2': acc_2}

    def validation_step(self, batch):
        images, labels = batch 
        out1, out2, out = self(images)
        loss_1 = F.cross_entropy(out1, labels[0])
        loss_2 = F.cross_entropy(out2, labels[1])
        loss_3 = F.cross_entropy(out, labels[2])
        loss = loss_1 + loss_2 + loss_3
        _, preds_1 = torch.max(out1, dim=-1)
        _, labels_1 = torch.max(labels[0], dim=-1)
        _, preds_2 = torch.max(out2, dim=-1)
        _, labels_2 = torch.max(labels[1], dim=-1)
        _, preds = torch.max(out, dim=-1)
        _, labels_ = torch.max(labels[2], dim=-1)
        acc_1 = torch.tensor(torch.sum(preds_1==labels_1).item() / (len(preds_1)*27))
        acc_2 = torch.tensor(torch.sum(preds_2==labels_2).item() / (len(preds_1)*27))
        acc = torch.tensor(torch.sum(preds==labels_).item() / (len(preds)*54))
        # with torch.no_grad():
        #     self.acc_gap = 1 - torch.tensor([acc_1, acc_2, acc])
        return {'val_loss': loss, 'val_acc': acc, 'val_acc_1': acc_1, 'val_acc_2': acc_2}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f} val_acc: {:.4f} acc_1: {:.4f} acc_2: {:.4f} val_acc_1: {:.4f} val_acc_2: {:.4f}".format(
            epoch, result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc'], result['acc_1'], result['acc_2'], result['val_acc_1'], result['val_acc_2']))

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        batch_accs_1 = [x['val_acc_1'] for x in outputs]
        epoch_acc_1 = torch.stack(batch_accs_1).mean()
        batch_accs_2 = [x['val_acc_2'] for x in outputs]
        epoch_acc_2 = torch.stack(batch_accs_2).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), 'val_acc_1': epoch_acc_1.item(), 'val_acc_2': epoch_acc_2.item()}

    def predict(self, images):
        outputs = self(images)
        _, preds = torch.max(torch.sum(outputs, dim=0), dim=-1)
        return preds


# class RubiksCubeModel(Module):
#     def __init__(self):
#         super(RubiksCubeModel, self).__init__()
#         self.conv1 = Conv2d(3, 8, 3)
#         self.conv2 = Conv2d(8, 16, 3)
#         self.conv3 = Conv2d(16, 32, 3)
#         self.conv4 = Conv2d(32, 64, 3)
#         self.conv5 = Conv2d(64, 128, 3)
#         self.conv6 = Conv2d(128, 256, 3)
#         self.conv7 = Conv2d(256, 512, 2)
#         self.conv8 = Conv2d(512, 1024, 2)
#         self.batch_norm1 = torch.nn.BatchNorm2d(8)
#         self.batch_norm2 = torch.nn.BatchNorm2d(16)
#         self.batch_norm3 = torch.nn.BatchNorm2d(32)
#         self.batch_norm4 = torch.nn.BatchNorm2d(64)
#         self.batch_norm5 = torch.nn.BatchNorm2d(128)
#         self.batch_norm6 = torch.nn.BatchNorm2d(256)
#         self.batch_norm7 = torch.nn.BatchNorm2d(512)
#         self.batch_norm8 = torch.nn.BatchNorm2d(1024)
#         self.layernorm1 = torch.nn.LayerNorm((8, 47, 47))
#         self.layernorm2 = torch.nn.LayerNorm((16, 22, 22))
#         self.layernorm3 = torch.nn.LayerNorm((32, 20, 20))
#         self.layernorm4 = torch.nn.LayerNorm((64, 18, 18))
#         self.layernorm5 = torch.nn.LayerNorm((128, 16, 16))
#         self.layernorm6 = torch.nn.LayerNorm((256, 14, 14))
#         self.layernorm7 = torch.nn.LayerNorm((512, 6, 6))
#         self.layernorm8 = torch.nn.LayerNorm((1024, 2, 2))
#         self.pool = MaxPool2d(2, 2)
#         self.fc1 = Linear(1024 * 2 * 2, 1024)
#         self.fc2 = Linear(1024, 2048)
#         self.fc3 = Linear(2048, 1024)
#         self.fc4 = Linear(1024, 27 * 6)
#         self.relu = ReLU()
#         self.dropout1 = Dropout(0.02)
#         self.dropout2 = Dropout(0.02)

#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x))) # (3, 96, 96) -> (8, 94, 94) -> (8, 47, 47)
#         x = self.batch_norm1(x) # (8, 47, 47) -> (8, 47, 47)
#         x = self.layernorm1(x)  # (8, 47, 47) -> (8, 47, 47)
#         x = self.pool(self.relu(self.conv2(x))) # (8, 47, 47) -> (16, 45, 45) -> (16, 22, 22)
#         x = self.batch_norm2(x) # (16, 22, 22) -> (16, 22, 22)
#         x = self.layernorm2(x)  # (16, 22, 22) -> (16, 22, 22)
#         x = self.relu(self.conv3(x)) # (16, 22, 22) -> (32, 20, 20)
#         x = self.batch_norm3(x) # (32, 20, 20) -> (32, 20, 20)
#         x = self.layernorm3(x)  # (32, 20, 20) -> (32, 20, 20)
#         x = self.relu(self.conv4(x)) # (32, 20, 20) -> (64, 18, 18)
#         x = self.batch_norm4(x) # (64, 18, 18) -> (64, 18, 18)
#         x = self.layernorm4(x)  # (64, 18, 18) -> (64, 18, 18)
#         x = self.relu(self.conv5(x)) # (64, 18, 18) -> (128, 16, 16)
#         x = self.batch_norm5(x) # (128, 16, 16)-> (128, 16, 16)
#         x = self.layernorm5(x)  # (128, 16, 16) -> (128, 16, 16)
#         x = self.relu(self.conv6(x)) # (128, 16, 16) -> (256, 14, 14)
#         x = self.batch_norm6(x) # (256, 14, 14) -> (256, 14, 14)
#         x = self.layernorm6(x)  # (256, 14, 14) -> (256, 14, 14)
#         x = self.pool(self.relu(self.conv7(x))) # (256, 14, 14) -> (512, 12, 12) -> (512, 6, 6)
#         x = self.batch_norm7(x) # (512, 6, 6) -> (512, 6, 6)
#         x = self.layernorm7(x)  # (512, 6, 6) -> (512, 6, 6)
#         x = self.pool(self.relu(self.conv8(x))) # (512, 6, 6) -> (1024, 4, 4) -> (1024, 2, 2)
#         x = self.batch_norm8(x) # (1024, 2, 2) -> (1024, 2, 2)
#         x = self.layernorm8(x)  # (1024, 2, 2) -> (1024, 2, 2)
#         x = x.view(-1, 1024 * 2 * 2) # (1024, 2, 2) -> (4096)
#         x = self.dropout1(self.relu(self.fc1(x)))
#         x = self.dropout2(self.relu(self.fc2(x)))
#         x = self.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x.view(-1, 27, 6)

#     def training_step(self, batch):
#         images, labels = batch 
#         out = self(images)
#         loss = F.cross_entropy(out, labels)
#         _, preds = torch.max(out, dim=-1)
#         _, labels = torch.max(labels, dim=-1)
#         acc = torch.tensor(torch.sum(preds==labels).item() / (len(preds)*27))
#         return {'loss': loss, 'acc': acc}

#     def validation_step(self, batch):
#         images, labels = batch 
#         out = self(images)  
#         loss = F.cross_entropy(out, labels)
#         _, preds = torch.max(out, dim=-1)
#         _, labels = torch.max(labels, dim=-1)
#         acc = torch.tensor(torch.sum(preds == labels).item() / (len(preds)*27))
#         return {'val_loss': loss.detach(), 'val_acc': acc}

#     def epoch_end(self, epoch, result):
#         print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f} val_acc: {:.4f}".format(
#             epoch, result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc']))

#     def validation_epoch_end(self, outputs):
#         batch_losses = [x['val_loss'] for x in outputs]
#         epoch_loss = torch.stack(batch_losses).mean()
#         batch_accs = [x['val_acc'] for x in outputs]
#         epoch_acc = torch.stack(batch_accs).mean()
#         return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

#     def predict(self, images):
#         outputs = self(images)
#         _, preds = torch.max(torch.sum(outputs, dim=0), dim=-1)
#         return preds