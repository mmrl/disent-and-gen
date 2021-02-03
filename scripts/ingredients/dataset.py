"""
Sacred Ingredient for datasets

This ingredient has the functions to load datsets and DataLoaders for training.
Selecting a dataset is a matter of passing the corresponding name. There is a
function to get the splits, and one to show them (assuming they are iamges).

Three datasets are currently supported, dSprites, 3DShapes and MPI3D. The
transformation dataset can also be loaded using this function.
"""


import sys
import torch
from torch.utils.data import DataLoader
from sacred import Ingredient

import matplotlib.pyplot as plt

if sys.path[0] != '../src':
    sys.path.insert(0, '../src')

# from dataset.tdisc import load_tdata
from dataset.sprites import load_sprites
from dataset.shapes3d import load_shapes3d
from dataset.mpi import load_mpi3d
from dataset.transforms import Triplets

import configs.datasplits as splits


dataset = Ingredient('dataset')
load_sprites = dataset.capture(load_sprites)
load_shapes3d = dataset.capture(load_shapes3d)
load_mpi3d = dataset.capture(load_mpi3d)
load_composition = dataset.capture(Triplets)

dataset.add_config(setting='unsupervised')
dataset.add_named_config('unsupervised', setting='unsupervised')
dataset.add_named_config('supervised', setting='supervised')


@dataset.capture
def get_dataset(dataset):
    if dataset == 'dsprites':
        dataset_loader = load_sprites
    elif dataset == 'shapes3d':
        dataset_loader = load_shapes3d
    elif dataset == 'mpi3d':
        dataset_loader = load_mpi3d
    elif dataset == 'composition':
        dataset_loader = load_composition
    else:
        raise ValueError('Unrecognized dataset {}'.format(dataset))

    return dataset_loader


@dataset.capture
def init_loader(dataset, batch_size, **loader_kwargs):
    kwargs = {'shuffle': True, 'pin_memory': True, 'prefetch_factor': 2,
              'num_workers': 4, 'persistent_workers': False}
    kwargs.update(**loader_kwargs)

    kwargs['pin_memory'] = kwargs['pin_memory'] and torch.cuda.is_available()
    loader = DataLoader(dataset, batch_size, **kwargs)

    return loader


@dataset.command(unobserved=True)
def plot():
    dataset = get_dataset()(setting='supervised')
    loader = init_loader(dataset, 1, pin_memory=False,
                         shuffle=False, n_workers=1)

    for img, t in loader:
        img = img.reshape(loader.dataset.img_size)
        img = img.squeeze().numpy()

        if len(img.shape) == 3:
            img = img.transpose(1, 2, 0)
            cmap = None
        else:
            cmap = 'Greys_r'

        plt.imshow(img, cmap=cmap, vmin=0, vmax=1)
        plt.show(block=True)


@dataset.capture
def get_data_spliters(dataset, condition=None, variant=None):
    if condition is None:
        return None, None
    if dataset == 'dsprites':
        return splits.Dsprites.get_splits(condition, variant)
    elif dataset == 'shapes3d':
        return splits.Shapes3D.get_splits(condition, variant)
    elif dataset == 'mpi3d':
        return splits.MPI3D.get_splits(condition, variant)
    else:
        raise ValueError('Condition given,'
                         'but dataset {} is invalid'.format(dataset))
