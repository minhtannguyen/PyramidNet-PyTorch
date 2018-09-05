# coding: utf-8

import numpy as np

import torch
import torch.utils.data

import torchvision
import torchvision.models
import torchvision.transforms

import transforms

def get_loader(batch_size, num_workers, config):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(probability = config['erase_prob'], sh = config['erase_sh'], r1 = config['erase_r1'], ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if config['dataset'] == 'cifar10':
        dataset_dir = '~/.torchvision/datasets/CIFAR10'
        train_dataset = torchvision.datasets.CIFAR10(
            dataset_dir, train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            dataset_dir, train=False, transform=test_transform, download=True)
    elif config['dataset'] == 'cifar100':
        dataset_dir = '~/.torchvision/datasets/CIFAR100'
        train_dataset = torchvision.datasets.CIFAR100(
            dataset_dir, train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR100(
            dataset_dir, train=False, transform=test_transform, download=True)
    else:
        print('dataset is not available')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader