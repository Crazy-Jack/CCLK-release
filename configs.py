import os
import json

import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

from augmentation import ColourDistortion
from dataset import *
from models import *


def get_datasets(dataset, augment_clf_train=False, add_indices_to_data=False, num_positive=None, attributes=False, num_attr=-1, no_color_distor=False):

    CACHED_MEAN_STD = {
        'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        'cifar100': ((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
        'stl10': ((0.4409, 0.4279, 0.3868), (0.2309, 0.2262, 0.2237)),
        'imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        'imagenet100-clip': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        'imagenet100': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        'utzappos': ((0.8342, 0.8142, 0.8081), (0.2804, 0.3014, 0.3072)),
        'cub': ((0.4863, 0.4999, 0.4312), (0.2070, 0.2018, 0.2428)),
    }

    PATHS = {
    }
    try:
        with open('dataset-paths.json', 'r') as f:
            local_paths = json.load(f)
            PATHS.update(local_paths)
    except FileNotFoundError:
        pass
    root = PATHS[dataset]

    #################
    PATHS_condition_continuous_feat = {
    }

    try:
        with open("condition_continuous_feat.json", 'r') as f:
            local_paths = json.load(f)
            PATHS_condition_continuous_feat.update(local_paths)
    except FileNotFoundError:
        pass

    #################



    # Data
    if dataset == 'stl10':
        img_size = 96
    elif dataset in ['imagenet', 'imagenet100', 'imagenet100-clip', 'wider', 'cub']:
        img_size = 224
    elif dataset in ['cifar10', 'cifar100', 'utzappos']:
        img_size = 32
    elif dataset in ['colorMNIST']:
        img_size = 32
    else:
        raise NotImplementedError

    if dataset == 'colorMNIST':
        if not no_color_distor:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                ColourDistortion(s=0.5),
                transforms.ToTensor(),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            ColourDistortion(s=0.5),
            transforms.ToTensor(),
            transforms.Normalize(*CACHED_MEAN_STD[dataset]),
        ])

    if dataset in ['imagenet', 'imagenet100-clip', 'imagenet100', 'wider', 'cub']:
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(*CACHED_MEAN_STD[dataset]),
        ])
    elif dataset in ['utzappos']:
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(*CACHED_MEAN_STD[dataset]),
        ])
    elif dataset in ['colorMNIST']:
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(*CACHED_MEAN_STD[dataset]),
        ])

    if augment_clf_train:
        if dataset == 'colorMNIST':
            transform_clftrain = transforms.Compose([
                transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform_clftrain = transforms.Compose([
                transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*CACHED_MEAN_STD[dataset]),
            ])
    else:
        transform_clftrain = transform_test

    if dataset == 'cifar100':
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.CIFAR100)
        else:
            dset = torchvision.datasets.CIFAR100
        if num_positive is None:
            trainset = CIFAR100Biaugment(root=root, train=True, download=True, transform=transform_train)
        else:
            trainset = CIFAR100Multiaugment(root=root, train=True, download=True, transform=transform_train,
                                            n_augmentations=num_positive)
        testset = dset(root=root, train=False, download=True, transform=transform_test)
        clftrainset = dset(root=root, train=True, download=True, transform=transform_clftrain)
        num_classes = 100
        stem = StemCIFAR
    elif dataset == 'colorMNIST':
        trainset = ColorMNISTBiAugDataset(root=root, split='train', transform=transform_train, biaug=True, conditional=True)
        clftrainset = ColorMNISTBiAugDataset(root=root, split='train', transform=transform_clftrain, biaug=False, conditional=False)
        clf_conditional_trainset = ColorMNISTBiAugDataset(root=root, split='train', transform=transform_clftrain, biaug=False, conditional=True)
        testset = ColorMNISTBiAugDataset(root=root, split='test', transform=transform_test, biaug=False, conditional=False)
        test_conditional_set = ColorMNISTBiAugDataset(root=root, split='test', transform=transform_test, biaug=False, conditional=True)
        testset = (testset, test_conditional_set)
        clftrainset = (clftrainset, clf_conditional_trainset)

        num_classes = 10
        stem = StemCIFAR

    elif dataset == 'cifar10':
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.CIFAR10)
        else:
            dset = torchvision.datasets.CIFAR10
        if num_positive is None:
            trainset = CIFAR10Biaugment(root=root, train=True, download=True, transform=transform_train)
        else:
            trainset = CIFAR10Multiaugment(root=root, train=True, download=True, transform=transform_train,
                                           n_augmentations=num_positive)
        testset = dset(root=root, train=False, download=True, transform=transform_test)
        clftrainset = dset(root=root, train=True, download=True, transform=transform_clftrain)
        num_classes = 10
        stem = StemCIFAR
    elif dataset == 'utzappos':
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.ImageFolder)
        else:
            dset = torchvision.datasets.ImageFolder

        if num_positive is None:
            trainset = ImageFolderBiaugment(root=os.path.join(root, 'train'), transform=transform_train)
        else:
            raise NotImplementedError
        testset = dset(root=os.path.join(root, 'val'), transform=transform_test)
        clftrainset = dset(root=os.path.join(root, 'train'), transform=transform_clftrain)
        num_classes = len(testset.classes)
        stem = StemCIFAR

    elif dataset == 'stl10':
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.STL10)
        else:
            dset = torchvision.datasets.STl10
        if num_positive is None:
            trainset = STL10Biaugment(root=root, split='unlabeled', download=True, transform=transform_train)
        else:
            raise NotImplementedError
        testset = dset(root=root, split='train', download=True, transform=transform_test)
        clftrainset = dset(root=root, split='test', download=True, transform=transform_clftrain)
        num_classes = 10
        stem = StemSTL
    elif dataset in ['imagenet', 'imagenet100', 'imagenet1k', 'wider', 'cub']:
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.ImageFolder)
        else:
            dset = torchvision.datasets.ImageFolder

        if num_positive is None:
            trainset = ImageFolderBiaugment(root=os.path.join(root, 'train'), transform=transform_train)
        else:
            raise NotImplementedError
        testset = dset(root=os.path.join(root, 'val'), transform=transform_test)
        clftrainset = dset(root=os.path.join(root, 'train'), transform=transform_clftrain)
        num_classes = len(testset.classes)
        stem = StemImageNet
    elif dataset in ['imagenet100-clip']:
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.ImageFolder)
        else:
            dset = torchvision.datasets.ImageFolder

        if num_positive is None:
            trainset = CKSDataset(transform=transform_train, pickle_file_path=PATHS_condition_continuous_feat[dataset], root=os.path.join(root, 'train'))
        else:
            raise NotImplementedError
        testset = dset(root=os.path.join(root, 'val'), transform=transform_test)
        clftrainset = dset(root=os.path.join(root, 'train'), transform=transform_clftrain)
        num_classes = len(testset.classes)
        stem = StemImageNet

    else:
        raise ValueError("Bad dataset value: {}".format(dataset))


    # handle attributes
    if attributes:

        PATHS_attribute_pd_path = {
        }

        PATHS_attribute_data_root_folder = {
        }
        try:
            with open('attribute_pd_paths.json', 'r') as f:
                local_paths = json.load(f)
                PATHS_attribute_pd_path.update(local_paths)
            with open('attribute_data_root_folders.json', 'r') as f:
                local_paths = json.load(f)
                PATHS_attribute_data_root_folder.update(local_paths)

        except FileNotFoundError:
            pass

        attribute_pd_path = PATHS_attribute_pd_path[dataset]
        attribute_data_root_folder = PATHS_attribute_data_root_folder[dataset]

        train_meta_df = pd.read_csv(attribute_pd_path, index_col=0)
        trainset = AttributeImageDataset(df=train_meta_df, data_path=attribute_data_root_folder, transform = transform_train, num_attr=num_attr)

    return trainset, testset, clftrainset, num_classes, stem
