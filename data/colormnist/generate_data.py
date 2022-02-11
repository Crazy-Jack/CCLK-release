from dataset import ColorMNIST
import numpy as np
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os

random_std = 40
num_color_together = 4
train_set = ColorMNIST('back', 'train', path='.', random_std=random_std, num_color_together=num_color_together, random_generate_color=True)
# train_set = ColorMNIST('back', 'train', path='.', random_std=random_std, num_color_together=num_color_together, random_generate_color=False)
train_loader = DataLoader(train_set, batch_size=64, shuffle=False)

train_augment_colors = []
train_images = []
train_labels = []
for image, label, color in tqdm(train_loader):
    # print(image.shape, label.shape, color.shape)
    train_augment_colors.append(color)
    train_images.append(image)
    train_labels.append(label)

train_augment_colors = torch.cat(train_augment_colors).numpy()
train_images = torch.cat(train_images).numpy()
train_labels = torch.cat(train_labels).numpy()
os.makedirs('colorMNIST/train', exist_ok=True)
print(f"TRAIN")
print(train_augment_colors.shape, train_images.shape, train_labels.shape)
np.save('colorMNIST/train/train_augment_colors.npy', train_augment_colors)
np.save('colorMNIST/train/train_images.npy', train_images)
np.save('colorMNIST/train/train_labels.npy', train_labels)



random_std = 40
num_color_together = 4
test_set = ColorMNIST('back', 'test', path='.', random_std=random_std, num_color_together=num_color_together, random_generate_color=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

test_augment_colors = []
test_images = []
test_labels = []
for image, label, color in tqdm(test_loader):
    # print(image.shape, label.shape, color.shape)
    test_augment_colors.append(color)
    test_images.append(image)
    test_labels.append(label)

test_augment_colors = torch.cat(test_augment_colors).numpy()
test_images = torch.cat(test_images).numpy()
test_labels = torch.cat(test_labels).numpy()
os.makedirs('colorMNIST/test', exist_ok=True)
print(f"Test... ")
print(test_augment_colors.shape, test_images.shape, test_labels.shape)
np.save('colorMNIST/test/test_augment_colors.npy', test_augment_colors)
np.save('colorMNIST/test/test_images.npy', test_images)
np.save('colorMNIST/test/test_labels.npy', test_labels)
