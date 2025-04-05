import os
import sys
sys.path.append("..")

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import ConcatDataset

from dataloader.CIR_data_utils import time_series_mixup, amplitude_intervention, phase_intervention
from utils import clip_probability, adjust_probability
from utils import ImbalancedDatasetSampler, DICISampler, DomainClassSampler, DomainSizeClassSampler


class RawLoadDataset(Dataset):
    def __init__(self, dataset):
        super(RawLoadDataset, self).__init__()
        x = dataset['samples']
        label = dataset['labels']

        if len(x.shape) < 3:
            x = x.unsqueeze(2)

        if x.shape.index(min(x.shape)) != 1:  # make sure the Channels in second dim
            x = x.permute(0, 2, 1)

        self.x_data = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        self.label_data = torch.from_numpy(label) if isinstance(label, np.ndarray) else label

        if self.label_data.dim() > 1:
            self.label_data = self.label_data.squeeze(1)

        self.len = x.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.label_data[index]

    def __len__(self):
        return self.len

    def get_labels(self):
        return list(self.label_data)


class LoadClassDomainDataset(Dataset):
    def __init__(self, dataset):
        super(LoadClassDomainDataset, self).__init__()
        x = dataset['samples']
        label = dataset['labels']
        domain_label = dataset['domain_labels']

        if len(x.shape) < 3:
            x = x.unsqueeze(2)

        if x.shape.index(min(x.shape)) != 1:  # make sure the Channels in second dim
            x = x.permute(0, 2, 1)

        self.x_data = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        self.label_data = torch.from_numpy(label) if isinstance(label, np.ndarray) else label
        self.domain_label_data = torch.from_numpy(domain_label) if isinstance(domain_label,
                                                                              np.ndarray) else domain_label

        if self.label_data.dim() > 1:
            self.label_data = self.label_data.squeeze(1)

        if self.domain_label_data.dim() > 1:
            self.domain_label_data = self.domain_label_data.squeeze(1)

        self.len = x.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.label_data[index], self.domain_label_data[index]

    def __len__(self):
        return self.len

    def get_labels(self):
        return list(self.label_data), list(self.domain_label_data)


class LoadDataset(Dataset):
    def __init__(self, dataset, std):
        super(LoadDataset, self).__init__()
        x = dataset['samples']
        label = dataset['labels']
        entropy = dataset['probability']
        self.std = std

        if len(x.shape) < 3:
            x = x.unsqueeze(2)

        if x.shape.index(min(x.shape)) != 1:  # make sure the Channels in second dim
            x = x.permute(0, 2, 1)

        self.x_data = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        self.label_data = torch.from_numpy(label) if isinstance(label, np.ndarray) else label
        self.entropy = torch.from_numpy(entropy) if isinstance(entropy, np.ndarray) else entropy

        if self.label_data.dim() > 1:
            self.label_data = self.label_data.squeeze(1)

        self.len = x.shape[0]

    def __getitem__(self, index):
        anchor, a_label = self.x_data[index], self.label_data[index].item()

        positive_indices = np.where(self.label_data == a_label)[0]
        positive_indices = positive_indices[positive_indices != index]
        positive_entropies = 1 - self.entropy[positive_indices, int(a_label)].numpy()
        positive_entropies = clip_probability(positive_entropies, 0.75)
        # print(len(positive_entropies), positive_entropies.sum())

        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(positive_indices, p=positive_entropies / positive_entropies.sum())

        negative_indices = np.where(self.label_data != a_label)[0]
        negative_entropies = self.entropy[negative_indices, int(a_label)].numpy()
        negative_entropies = clip_probability(negative_entropies, 0.75)
        # print(len(negative_entropies), negative_entropies.sum())
        negative_index = np.random.choice(negative_indices, p=negative_entropies / negative_entropies.sum())
        positive = self.x_data[positive_index]
        negative = self.x_data[negative_index]

        aug_x_data = amplitude_intervention(anchor, std=self.std)
        # aug_x_data = phase_intervention(anchor, std=self.std)
        # aug_x_data = anchor + torch.normal(mean=0, std=self.std, size=anchor.shape)
        aug_idx = np.random.choice(list(range(self.len)))
        x_data_aug = self.x_data[aug_idx]
        # aug_x_data, _ = time_series_mixup(x_data, x_data_aug)
        return anchor, aug_x_data, positive, negative, self.label_data[index]

    def __len__(self):
        return self.len

    def get_labels(self):
        return list(self.label_data)


# Imbalanced data sampler
def dataloader(configs, test, std=0.01):
    train_data = torch.load(os.path.join(configs.data_path, f'{configs.aim}_for_{test}.pt'))  # _DGFD, _SIDGFD
    val_data = torch.load(os.path.join(configs.data_path, f'{configs.aim}_{test}.pt'))  # _DGFD, _SIDGFD
    # HVAC: {configs.aim}_{test}, BGM: {configs.aim}_for_{test}_test
    test_data = torch.load(os.path.join(configs.data_path, f'{configs.aim}_{test}.pt'))

    train_data = LoadDataset(train_data, std)
    labels = np.array(train_data.get_labels())
    val_data = RawLoadDataset(val_data)
    test_data = RawLoadDataset(test_data)
    train_loader = DataLoader(dataset=train_data, batch_size=configs.batch_size, shuffle=False,
                              sampler=ImbalancedDatasetSampler(train_data, labels),
                              num_workers=32, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=configs.batch_size, shuffle=False, num_workers=0,
                            pin_memory=True, drop_last=False)
    test_loader = DataLoader(dataset=test_data, batch_size=configs.batch_size, shuffle=False, num_workers=0,
                             pin_memory=True, drop_last=False)

    return train_loader, val_loader, test_loader


# Load randomly shuffled data
def imb_dataloader(configs, trains, test):
    # print(os.getcwd())
    all_train_data = []
    for i in trains:
        train_data_i = torch.load(os.path.join(configs.data_path, f'{configs.aim}_{i}.pt'))
        train_data_i = LoadClassDomainDataset(train_data_i)
        all_train_data.append(train_data_i)
    val_data = torch.load(os.path.join(configs.data_path, f'{configs.aim}_{test}.pt'))  # _DGFD
    # HVAC: {configs.aim}_{test}, BGM: {configs.aim}_for_{test}_test
    test_data = torch.load(os.path.join(configs.data_path, f'{configs.aim}_{test}.pt'))

    train_data = ConcatDataset(all_train_data)
    val_data = RawLoadDataset(val_data)
    test_data = RawLoadDataset(test_data)

    train_loader = DataLoader(dataset=train_data, batch_size=configs.batch_size, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=configs.batch_size, shuffle=False, num_workers=0,
                            pin_memory=True, drop_last=False)
    test_loader = DataLoader(dataset=test_data, batch_size=configs.batch_size, shuffle=False, num_workers=0,
                             pin_memory=True, drop_last=False)

    return train_loader, val_loader, test_loader

