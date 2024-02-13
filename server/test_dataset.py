from dataset import Dataset
from torch.utils.data import DataLoader, random_split
import torch
import os

model_type = 'pose_estimation'
dataset = Dataset(model_type=model_type)

num_examples = len([f for f in os.listdir(model_type) if f.endswith('npy')])
assert dataset.size == num_examples, f'Expected {num_examples}, got {dataset.size}'
assert len(dataset) == num_examples, f'Expected {num_examples}, got {len(dataset)}'

expected_len = 2
example_0 = dataset[0]
assert len(example_0) == expected_len, f'Expected {expected_len}, got {len(example_0)}'
images, state = example_0

expected_len = 2 if model_type == 'color_recognition' else 1
assert len(images) == expected_len, f'Expected {expected_len}, got {len(images)}'

expected_shape = (6, 3, 3) if model_type == 'color_recognition' else (1, 6)
assert state.shape == expected_shape, f'Expected {expected_shape}, got {state.shape}'
assert type(images[0]) == torch.Tensor, f'Expected torch.Tensor, got {type(images[0])}'

img_shape = images[0].shape

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

assert len(train_ds) == train_size
assert len(val_ds) == val_size

batch_size = 8

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=28, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=28, pin_memory=True)

expected_img_shape = (batch_size, *img_shape)
expected_state_shape = (batch_size, *expected_shape)
for batch in train_loader:
    assert len(batch) == 2, f'Expected 2, got {len(batch)}'
    images, state = batch
    assert images[0].shape == expected_img_shape, f'Expected {expected_img_shape}, got {tuple(images[0].shape)}'
    assert state.shape == expected_state_shape, f'Expected {expected_state_shape}, got {state.shape}'
    break

print('Dataset: All tests passed!')