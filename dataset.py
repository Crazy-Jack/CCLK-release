import numpy as np
import torch
import torchvision
from PIL import Image
import os
from copy import deepcopy
import pickle


class CIFAR10Biaugment(torchvision.datasets.CIFAR10):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pil_img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(pil_img)
            img2 = self.transform(pil_img)
        else:
            img2 = img = pil_img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, img2), target, index


class CIFAR100Biaugment(CIFAR10Biaugment):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10Biaugment` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class STL10Biaugment(torchvision.datasets.STL10):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(pil_img)
            img2 = self.transform(pil_img)
        else:
            img2 = img = pil_img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, img2), target, index


class CIFAR10Multiaugment(torchvision.datasets.CIFAR10):

    def __init__(self, *args, n_augmentations=8, **kwargs):
        super(CIFAR10Multiaugment, self).__init__(*args, **kwargs)
        self.n_augmentations = n_augmentations
        assert self.transforms is not None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pil_img = Image.fromarray(img)

        imgs = [self.transform(pil_img) for _ in range(self.n_augmentations)]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return torch.stack(imgs, dim=0), target, index


class CIFAR100Multiaugment(CIFAR10Multiaugment):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10Biaugment` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class ImageNetBiaugment(torchvision.datasets.ImageNet):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            img = self.transform(sample)
            img2 = self.transform(sample)
        else:
            img2 = img = sample
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, img2), target, index

class ImageFolderBiaugment(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            img = self.transform(sample)
            img2 = self.transform(sample)
        else:
            img2 = img = sample
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, img2), target, index


class SideInfomation:
    def __init__(self, df_path):
        self.df = pd.read_csv(df_path)



class AttributeImageDataset(torch.utils.data.Dataset):
    '''
    dataset for attribute neural selection
    '''
    def __init__(self, df, data_path, transform=None, num_attr=-1):
        '''
        label_known_mask: str, the column name of mask for whether the class labels are known
        '''
        self.df = df
        self.attr_list = [i for i in self.df.columns if 'attr_val' in i]
        self.attr_num = len(self.attr_list)
        self.transform = transform
        self.data_path = data_path
        self.num_attr = num_attr

        self.attr_list = self.rank_attributes(self.attr_list)
        if self.num_attr > 0:
            self.attr_list = self.attr_list[:self.num_attr]
        if -1 in self.df.index:
            self.df = df.drop([-1])

    def rank_attributes(self, attr_list, method_='entropy'):
        data = self.df.loc[:, attr_list]
        if method_ == 'entropy':
            entropy_attr = data.apply(lambda col: entropy(col), axis=0) # should be [num_of_attr]
            sort_order = np.argsort(entropy_attr)[::-1]
            attr_list_new = []
            for i in sort_order:
                attr_list_new.append(attr_list[i])
        return attr_list

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        '''
        if self.label_known_mask is set:
            return attributes, class, mask
            Note: mask: 1 indicate known, 0 indicate unknown
                  class: -1 for unknown, otherwise known.
        else:
            return attributes, -1, 0
        '''
        attributes = np.array(self.df.iloc[index][self.attr_list].tolist(), dtype='float32')
        if self.num_attr >0:
            attributes = attributes[:self.num_attr]

        # img

        img = Image.open(os.path.join(self.data_path, self.df.iloc[index]['path']))
        img = deepcopy(img)
        if not img.mode == 'RGB':
            img = img.convert("RGB")

        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)


        return (img1, img2), attributes, index

# dataset for additional information conditioning
class CKSDataset(torch.utils.data.Dataset):
    """This dataset is particularly suitable for conditioning on additional continuous value
    user should provide the file path, as well as the corresponding continuous vector accosicated with each image
    """
    def __init__(self, transform=None, pickle_file_path="", root=None):
        super(CKSDataset, self).__init__()

        with open(pickle_file_path, 'rb') as f:
            feature_file = pickle.load(f)

        self.paths = feature_file[0] # list,
        self.features = feature_file[1].astype(np.float32) # numpy array, [num_data, feature_dim]
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # transform path
        if self.root:
            image_id = "/".join(self.paths[idx].split("/")[-2:])
            path = os.path.join(self.root, image_id)

        img = Image.open(path)
        img = deepcopy(img)
        if not img.mode == 'RGB':
            img = img.convert("RGB")

        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)
        else:
            img1 = img2 = img

        feature_idx = self.features[idx]

        return (img1, img2), feature_idx, idx

class ColorMNISTBiAugDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None, biaug=True, conditional=True, mask_lbl_percent=0.1):
        super(ColorMNISTBiAugDataset, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.biaug = biaug
        self.conditional = conditional
        # train
        self.img = np.load(os.path.join(root, split, f'{split}_images.npy'))
        self.label = np.load(os.path.join(root, split, f'{split}_labels.npy'))
        self.color = np.load(os.path.join(root, split, f'{split}_augment_colors.npy'))


        assert self.img.shape[0] == self.label.shape[0]
        assert self.img.shape[0] == self.color.shape[0]

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, idx):
        img = (self.img[idx] * 255).astype(np.uint8).transpose([1, 2, 0])
        img = Image.fromarray(img)
        lbl = self.label[idx]
        color = self.color[idx].astype(np.float32)

        if self.transform:
            img1 = self.transform(img)
            if self.biaug:
                img2 = self.transform(img)
                img = (img1, img2)
            else:
                img = img1

        if self.conditional:
            return img, lbl, torch.from_numpy(color)
        else:
            return img, lbl

# helper functions

def entropy(x):
    '''
    H(x)
    '''
    unique, count = np.unique(x, return_counts=True, axis=0)
    prob = count/len(x)
    H = np.sum((-1)*prob*np.log2(prob))

    return H

def add_indices(dataset_cls):
    class NewClass(dataset_cls):
        def __getitem__(self, item):
            output = super(NewClass, self).__getitem__(item)
            return (*output, item)

    return NewClass



