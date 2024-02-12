from model import SingleImageModel, ColorRecognizer
from dataset import Dataset
from torch.utils.data import DataLoader, random_split
import torch

resolution = (96, 96)
input_shape = (3, *resolution)

split_ratio = 0.8

single_image_model = SingleImageModel(input_shape=input_shape)
model = ColorRecognizer(input_shape=input_shape)
dataset = Dataset(resolution)

train_size = int(split_ratio * len(dataset))
val_size = len(dataset) - train_size

print(f"Train size: {train_size}, Val size: {val_size}", len(dataset))

train_ds, val_ds = random_split(dataset, [train_size, val_size])

batch_size = 2

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=28, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=28, pin_memory=True)

for batch in train_loader:
    images, state = batch
    print(state.shape)
    single_image_out = single_image_model(images[0])
    print(single_image_out.shape)
    out = model(images)
    print(out.shape)
    break

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=28, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=28, pin_memory=True)

epochs = 5
lr = 0.001
gamma = 0.999
optimizer = torch.optim.Adam(model.parameters(), lr)

for epoch in range(epochs):
    for batch in train_loader:
        model.train()
        model.zero_grad()
        out = model.training_step(batch)
        loss = out['loss']
        loss.backward()
        optimizer.step()
        print(out)

    for batch in val_loader:
        model.eval()
        out = model.validation_step(batch)
        print(out)



