import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

import os
import torch.utils.data as data
from PIL import Image
import random

class IndexedDataset(data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        img, label = self.dataset[index]

        return img, label, index

    def __len__(self):
        return len(self.dataset)

def get_dataset(name,
                data_dir,
                size=64,
                transform=None):
                
    transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]) if transform == None else transform

    if name == 'image':
        print('Using image labels')
        dataset = datasets.ImageFolder(data_dir, transform)
        nlabels = len(dataset.classes)
    elif name == 'mnist':
        dataset = datasets.MNIST(
            root=data_dir, 
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])
        )
        nlabels = 10
    else:
        raise NotImplemented

    dataset = IndexedDataset(dataset)  # provide data with indexes 

    return dataset, nlabels