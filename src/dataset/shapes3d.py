"""
3DShapes dataset module

The module contains the code for loading the 3DShapes dataset. The dataset can
be loaded in 3 modes: supervised, unsupervised, and ground-truth latent
reconstruction. We mostly use the last for 3 training and the first one for
analyzing the results. Data loading of the batches is handled in the
corresponding Sacred ingredient.

The original dataset can be found at:
    https://github.com/deepmind/3d-shapes
"""


import numpy as np
import h5py
import torch
import torchvision.transforms as trans
from skimage.color import rgb2hsv
from torch.utils.data import Dataset
# from functools import partialmethod


def load_raw(path, latent_filter=None):
    data_zip = h5py.File(path, 'r')

    imgs = data_zip['images'][()]
    latent_values = data_zip['labels'][()]
    latent_classes = latent_values

    if latent_filter is not None:
        idx = latent_filter(latent_values, latent_classes)

        imgs = imgs[idx]
        latent_values = latent_values[idx]
        latent_classes = latent_classes[idx]

    return imgs, latent_values, latent_classes


class Shapes3D(Dataset):
    """
    Disentangled dataset used in Kim and Mnih, (2019)

    #==========================================================================
    # Latent Dimension,    Latent values                                 N vals
    #==========================================================================

    # floor hue:           uniform in range [0.0, 1.0)                      10
    # wall hue:            uniform in range [0.0, 1.0)                      10
    # object hue:          uniform in range [0.0, 1.0)                      10
    # scale:               uniform in range [0.75, 1.25]                     8
    # shape:               0=square, 1=cylinder, 2=sphere, 3=pill            4
    # orientation          uniform in range [-30, 30]                       15
    """
    def __init__(self, imgs, latent_values, latent_classes, color_mode='rgb',
                 transform=None, target_transform=None):
        self.imgs = imgs
        self.latent_values = latent_values
        self.latent_classes = latent_classes

        image_transforms = [trans.ToTensor(),
                            trans.ConvertImageDtype(torch.float32),
                            trans.Lambda(lambda x: x.flatten())]

        if color_mode == 'hsv':
            image_transforms.insert(0, trans.Lambda(rgb2hsv))

        latent_transforms = [trans.Lambda(lambda x: torch.from_numpy(x).to(
                                 dtype=torch.float32))]

        self.transform = trans.Compose(image_transforms)
        self.target_transform = trans.Compose(latent_transforms)

    def __getitem__(self, key):
        return (self.imgs[key],
                self.latent_values[key],
                self.latent_classes[key])

    def __len__(self):
        return len(self.imgs)

    files = {"train": "../data/raw/shapes3d/3dshapes.h5"}
    n_gen_factors = 6
    lat_names = ('floor_hue', 'wall_hue', 'object_hue',
                 'scale', 'shape', 'orientation')
    lat_sizes = np.array([10, 10, 10, 8, 4, 15])

    img_size = (3, 64, 64)

    lat_values = {'floor_hue': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  'wall_hue': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  'object_hue': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  'scale': [0.75, 0.82142857, 0.89285714, 0.96428571,
                            1.03571429, 1.10714286, 1.17857143, 1.25],
                  'shape': [0, 1, 2, 3],
                  'orientation': [-30., -25.71428571, -21.42857143,
                                  -17.14285714, -12.85714286, -8.57142857,
                                  -4.28571429, 0., 4.28571429, 8.57142857,
                                  12.85714286, 17.14285714, 21.42857143,
                                  25.71428571,  30.]}


class Unsupervised(Shapes3D):
    def __getitem__(self, key):
        imgs = self.transform(self.imgs[key])
        return imgs, imgs


class Supervised(Shapes3D):
    def __getitem__(self, idx):
        imgs, latent_values = self.imgs[idx], self.latent_values[idx]

        imgs = self.transform(imgs)
        classes = self.target_transform(latent_values)

        return imgs, classes


class Reconstruction(Shapes3D):
    def __getitem__(self, idx):
        imgs = self.imgs[idx]
        latent_values = self.latent_values[idx]

        imgs = self.transform(imgs)
        latent_values = self.target_transform(latent_values)

        return latent_values, imgs



def get_dataset_constr(setting):
    if setting == 'unsupervised':
        return Unsupervised
    elif setting == 'supervised':
        return Supervised
    elif setting == 'recons':
        return Reconstruction
    raise ValueError('Unrecognized setting "{}"'.format(setting))


def load_shapes3d(setting, data_filters=(None, None), decision=None,
                  color_mode='rgb', train=True, path=None):

    train_filter, test_filter = data_filters

    if path is None:
        path=Shapes3D.files['train']

    dataset_constr = get_dataset_constr(setting)
    if train:
        data = dataset_constr(*load_raw(path, train_filter),
                              target_transform=decision,
                              color_mode=color_mode)
    else:
        data = dataset_constr(*load_raw(path, test_filter),
                              target_transform=decision,
                              color_mode=color_mode)

    return data
