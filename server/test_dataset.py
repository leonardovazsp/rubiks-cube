from dataset import Dataset
from torch.utils.data import DataLoader, random_split
import torch
import os

dataset = Dataset()

num_examples = len(os.listdir('data')) // 3
assert dataset.size == num_examples
assert len(dataset) == num_examples

example_0 = dataset[0]
assert len(example_0) == 2, f'Expected 2, got {len(example_0)}'
images, state = example_0

assert len(images) == 2, f'Expected 2, got {len(images)}'
assert state.shape == (6, 3, 3), f'Expected (6, 3, 3), got {state.shape}'
assert type(images[0]) == type(images[1]) == torch.Tensor, f'Expected torch.Tensor, got {type(images[0])}, {type(images[1])}'

img_shape = images[0].shape

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

assert len(train_ds) == train_size
assert len(val_ds) == val_size

batch_size = 8

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=28, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=28, pin_memory=True)

for batch in train_loader:
    assert len(batch) == 2, f'Expected 2, got {len(batch)}'
    images, state = batch
    assert images[0].shape == (batch_size, *img_shape), f'Expected {(batch_size, *img_shape)}, got {tuple(images[0].shape)}'
    assert images[1].shape == (batch_size, *img_shape), f'Expected {(batch_size, *img_shape)}, got {tuple(images[1].shape)}'
    assert state.shape == (batch_size, 6, 3, 3), f'Expected ({batch_size}, 6, 3, 3), got {state.shape}'
    break

print('Dataset: All tests passed!')