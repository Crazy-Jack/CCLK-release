import os
import struct
import numpy as np


import os
import struct
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import matplotlib.pyplot as plt
__all__ = ["ColorMNIST"]


class ColorMNIST(Dataset):
    def __init__(self, color, split, path, transform_list=[], random_std=10, num_color_together=2, random_generate_color=False):
        assert color in ['num', 'back', 'both'], "color must be either 'num', 'back' or 'both"
        self.pallette = [[31, 119, 180],
                         [255, 127, 14],
                         [44, 160, 44],
                         [214, 39, 40],
                         [148, 103, 189],
                         [140, 86, 75],
                         [227, 119, 194],
                         [127, 127, 127],
                         [188, 189, 34],
                         [23, 190, 207]]

        if split == 'train':
            fimages = os.path.join(path, 'samples', 'train-images-idx3-ubyte')
            flabels = os.path.join(path, 'samples', 'train-labels-idx1-ubyte')
        else:
            fimages = os.path.join(path, 'samples', 't10k-images-idx3-ubyte')
            flabels = os.path.join(path, 'samples', 't10k-labels-idx1-ubyte')

        # Load images
        with open(fimages, 'rb') as f:
            _, _, rows, cols = struct.unpack(">IIII", f.read(16))
            self.images = np.fromfile(f, dtype=np.uint8).reshape(-1, rows, cols)

        # Load labels
        with open(flabels, 'rb') as f:
            struct.unpack(">II", f.read(8))
            self.labels = np.fromfile(f, dtype=np.int8)
            self.labels = torch.from_numpy(self.labels.astype(np.int))

        self.transform_list = transform_list
        self.color = color
        self.images = np.tile(self.images[:, :, :, np.newaxis], 3)
        
        self.random_std = random_std
        self.random_generate_color = random_generate_color # whether go completely random in the color
        self.num_color_together = num_color_together

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # Range [0,255]
        label = self.labels[idx]
        if self.random_generate_color:
            color_label = torch.randint(len(self.pallette), size=(1,)).item()
        else:
            color_label = label % self.num_color_together

        # Choose color
        if self.color == 'num':
            
            c = self.pallette[-(color_label + 1)] + (np.random.normal(size=3) * self.random_std).astype(int)
            c = np.clip(c, 0, 255)
            color = torch.from_numpy(c)
        elif self.color == 'back':
            c = self.pallette[color_label] + (np.random.normal(size=3) * self.random_std).astype(int)
            c = np.clip(c, 0, 255)
            color = torch.from_numpy(c)
        else:
            c = self.pallette[color_label] + (np.random.normal(size=3) * self.random_std).astype(int) 
            c = np.clip(c, 0, 255)
            c2 = self.pallette[-(color_label - 3)] + (np.random.normal(size=3) * self.random_std).astype(int) 
            c2 = np.clip(c2, 0, 255)
            color = torch.from_numpy(np.concatenate([c, c2]))
        # Assign color according to their class (0,10)
        if self.color == 'num':
            image[:, :, 0] = image[:, :, 0] / 255 * c[0]
            image[:, :, 1] = image[:, :, 1] / 255 * c[1]
            image[:, :, 2] = image[:, :, 2] / 255 * c[2]
        elif self.color == 'back':
            image[:, :, 0] = (255 - image[:, :, 0]) / 255 * c[0]
            image[:, :, 1] = (255 - image[:, :, 1]) / 255 * c[1]
            image[:, :, 2] = (255 - image[:, :, 2]) / 255 * c[2]
        else:
            image[:, :, 0] = image[:, :, 0] / 255 * c[0] + (255 - image[:, :, 0]) / 255 * c2[0]
            image[:, :, 1] = image[:, :, 1] / 255 * c[1] + (255 - image[:, :, 1]) / 255 * c2[1]
            image[:, :, 2] = image[:, :, 2] / 255 * c[2] + (255 - image[:, :, 2]) / 255 * c2[2]

        image = Image.fromarray(image)
        for t in self.transform_list:
            image = t(image)
        image = transforms.ToTensor()(image)  # Range [0,1]

        return image, label, color


if __name__ == '__main__':
    random_std = 50
    num_color_together = 4
    mydataset = ColorMNIST('back', 'train', path='.', random_std=random_std, num_color_together=num_color_together)

    num_instance = np.zeros(10)
    color_img_sets = [[] for _ in range(10)]
    i = 0
    collect_num = 20
    while num_instance.mean() < collect_num:
        img = mydataset[i][0].numpy().transpose([1, 2, 0])
        print(f"{i}: {img.shape}")
        lbl = mydataset[i][1]
        if num_instance[lbl] < collect_num:
            color_img_sets[lbl].append(img)
            num_instance[lbl] += 1
        i += 1

    x_all = np.vstack([np.hstack(k) for k in color_img_sets])
    plt.imsave(f"train_random_std_{random_std}.png", x_all)



    # test
    random_std = random_std
    num_color_together = 4
    mydataset = ColorMNIST('back', 'test', path='.', random_std=random_std, random_generate_color=True)

    num_instance = np.zeros(10)
    color_img_sets = [[] for _ in range(10)]
    i = 0
    collect_num = 20
    while num_instance.mean() < collect_num:
        img = mydataset[i][0].numpy().transpose([1, 2, 0])
        print(f"{i}: {img.shape}")
        lbl = mydataset[i][1]
        if num_instance[lbl] < collect_num:
            color_img_sets[lbl].append(img)
            num_instance[lbl] += 1
        i += 1

    x_all = np.vstack([np.hstack(k) for k in color_img_sets])
    plt.imsave(f"test_random_std_{random_std}.png", x_all)

