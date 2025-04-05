import numpy as np
import pandas as pd
from tsmoothie.utils_func import create_windows
from typing import Callable

import torch


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, labels: list = None, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def domain_label_reset(domain_labels):
    unique_values = torch.unique(domain_labels)
    value_map = {v.item(): idx for idx, v in enumerate(unique_values)}
    reset_labels = domain_labels.clone().apply_(lambda x: value_map[x])
    return reset_labels


def feature_masking(tensor1, tensor2, mask_ratio=0.2):
    # Generate a random mask with the same shape as input_tensor
    mask = torch.rand(tensor1.shape) > mask_ratio
    # Apply the mask to the input tensors
    masked_tensor1 = tensor1 * mask.float().to(tensor1.device)
    masked_tensor2 = tensor2 * mask.float().to(tensor2.device)
    return masked_tensor1, masked_tensor2


def clip_probability(probability, rate=0.8, epsilon=1e-20):
    zero_size = np.sum(probability == 0) / probability.size
    max_clipping_rate = max(zero_size, rate)
    min_clipping_rate = max(zero_size, 1 - rate)
    max_clipping = max(np.quantile(probability, max_clipping_rate), epsilon)
    min_clipping = max(np.quantile(probability, min_clipping_rate), epsilon)
    # Clip the values and add epsilon
    probability = np.clip(probability, a_min=min_clipping, a_max=max_clipping)
    e_probability = np.exp(probability/np.max(probability))
    probability = e_probability / e_probability.sum(axis=0)
    return probability


def adjust_probability(probability):
    # Calculate quartiles
    q1, q2, q3 = np.quantile(probability, [0.25, 0.5, 0.75])

    # Create an output array of the same shape as probability
    adjusted_probs = np.zeros_like(probability)

    # Adjust the probability values according to the quartile segment
    conditions = [probability <= q1,
                  (probability > q1) & (probability <= q2),
                  (probability > q2) & (probability <= q3),
                  probability > q3]
    values = [0.1, 0.2, 0.3, 0.4]

    for condition, value in zip(conditions, values):
        adjusted_probs[condition] = value

    return adjusted_probs

