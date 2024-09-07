import torch
import numpy as np

from os.path import join, dirname, realpath


default_cifar10_path = join(dirname(dirname(realpath(__file__))), 'shared_data', 'cifar10', 'full_set')

def load_dataset_to_torch(dataset_path = default_cifar10_path, test = True, train = True, reshape_and_divide = True):

    ret = []
    if test:
        test_images     = torch.tensor(np.load(join(dataset_path, 'test_images.npy')), dtype=torch.float32)
        if reshape_and_divide:
            test_images     = test_images.reshape(-1, 3, 32, 32)    # X
            test_images     = test_images.permute(0, 2, 3, 1) / 255.
        test_labels     = torch.tensor(np.load(join(dataset_path, 'test_labels.npy')), dtype=torch.float32)
        ret.extend([test_images, test_labels])
    if train:
        training_images = torch.tensor(np.load(join(dataset_path, 'training_images.npy')), dtype=torch.float32)
        if reshape_and_divide:
            training_images = training_images.reshape(-1, 3, 32, 32)
            training_images = training_images.permute(0, 2, 3, 1) / 255.
        training_labels = torch.tensor(np.load(join(dataset_path, 'training_labels.npy')), dtype=torch.float32)
        ret.extend([training_images, training_labels])

    ret.append(10)

    return tuple(ret)

