"""
dSprites dataset module

The module contains the code for loading the dSprites dataset. This dataset
contains transformations of simple sprites in 2 dimensions, which have no
detailed features.

The original dataset can be found at:
    https://github.com/deepmind/3d-shapes
"""


import numpy as np
import torch
import torchvision.transforms as trans
from torch.utils.data import Dataset, DataLoader, random_split
from functools import partialmethod


def partialclass(cls, *args, **kwds):

    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwds)

    return NewCls


def load_raw(path, latent_filter=None):
    data_zip = np.load(path, allow_pickle=True)

    imgs = data_zip['imgs'] * 255
    latent_values = data_zip['latents_values'][:, 1:]  # Remove luminescence
    latent_classes = data_zip['latents_classes'][:, 1:]

    if latent_filter is not None:
        idx = latent_filter(latent_values, latent_classes)

        imgs = imgs[idx]
        latent_values = latent_values[idx]
        latent_classes = latent_classes[idx]

    # imgs = torch.from_numpy(imgs).to(dtype=torch.float32)
    # latent_values = torch.from_numpy(latent_values).to(dtype=torch.float32)
    # latent_classes = torch.from_numpy(latent_classes).to(dtype=torch.float32)

    return imgs, latent_values, latent_classes


class Dsprites(Dataset):
    def __init__(self, imgs, latent_values, latent_classes,
                 transform=None, target_transform=None):
        self.imgs = imgs
        self.latent_values = latent_values
        self.latent_classes = latent_classes

        image_transforms = [trans.ToTensor(),
                            trans.ConvertImageDtype(torch.float32),
                            trans.Lambda(lambda x: x.flatten())]

        latent_transforms = [trans.Lambda(lambda x: torch.from_numpy(x).to(
                             dtype=torch.float32))]

        if transform is not None:
            image_transforms.append(transform)
        if target_transform is not None:
            image_transforms.append(target_transform)

        self.transform = trans.Compose(image_transforms)
        self.target_transform = trans.Compose(latent_transforms)

    def __getitem__(self, key):
        return (self.imgs[key], self.latent_values[key],
                self.latent_classes[key])

    def __len__(self):
        return len(self.imgs)

    urls = {"train": "https://github.com/deepmind/dsprites-dataset/blob/master/"
                     "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"}
    files = {"train": "../data/raw/dsprites/dsprite_train.npz"}
    lat_names = ('shape', 'scale', 'orientation', 'posX', 'posY')
    lat_sizes = np.array([1, 3, 6, 40, 32, 32])
    img_size = (1, 64, 64)
    n_gen_factors = 5
    background_color = np.zeros(3)
    lat_values = {'posX': np.array([0., 0.03225806, 0.06451613, 0.09677419,
                                    0.12903226, 0.16129032, 0.19354839,
                                    0.22580645, 0.25806452, 0.29032258,
                                    0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097,
                                    0.51612903, 0.5483871, 0.58064516,
                                    0.61290323, 0.64516129, 0.67741935,
                                    0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774,
                                    0.90322581, 0.93548387, 0.96774194, 1.]),
                  'posY': np.array([0., 0.03225806, 0.06451613, 0.09677419,
                                    0.12903226, 0.16129032, 0.19354839,
                                    0.22580645, 0.25806452, 0.29032258,
                                    0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097,
                                    0.51612903, 0.5483871, 0.58064516,
                                    0.61290323, 0.64516129, 0.67741935,
                                    0.70967742, 0.74193548, 0.77419355,
                                    0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774,
                                    0.90322581, 0.93548387, 0.96774194, 1.]),
                  'scale': np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
                  'orientation': np.array([0., 0.16110732, 0.32221463,
                                           0.48332195, 0.64442926, 0.80553658,
                                           0.96664389, 1.12775121, 1.28885852,
                                           1.44996584, 1.61107316, 1.77218047,
                                           1.93328779, 2.0943951, 2.25550242,
                                           2.41660973, 2.57771705, 2.73882436,
                                           2.89993168, 3.061039, 3.22214631,
                                           3.38325363, 3.54436094, 3.70546826,
                                           3.86657557, 4.02768289, 4.1887902,
                                           4.34989752, 4.51100484, 4.67211215,
                                           4.83321947, 4.99432678, 5.1554341,
                                           5.31654141, 5.47764873, 5.63875604,
                                           5.79986336, 5.96097068, 6.12207799,
                                           6.28318531]),
                  'shape': np.array([1., 2., 3.]),
                  'color': np.array([1.])}


class Unsupervised(Dsprites):
    def __getitem__(self, idx):
        imgs = self.transform(self.imgs[idx])
        return imgs, imgs


class Supervised(Dsprites):
    def __getitem__(self, idx):
        imgs, latent_values = self.imgs[idx], self.latent_values[idx]

        imgs = self.transform(imgs)
        classes = self.target_transform(latent_values)

        return imgs, classes


class SemiSupervised(Supervised):
    def __getitem__(self, idx):
        imgs, classes = super().__getitem__(idx)
        return imgs, (imgs, classes)


class Reconstruction(Dsprites):
    def __getitem__(self, idx):
        imgs = self.imgs[idx]
        latent_values = self.latent_values[idx]

        imgs = self.transform(imgs)
        latent_values = self.target_transform(latent_values)

        return latent_values, imgs


class SupervisedLatents(Dsprites):
    def __init__(self, decision, data, batch_size, shuffle=True):
        super().__init__(data, batch_size, shuffle)
        self.target_transform = trans.Lambda(decision)

    def __getitem__(self, idx):
        imgs, latent_values, latent_classes = super().__getitem__(idx)

        imgs = self.transform(imgs)
        classes = self.target_transform(latent_values, latent_classes)

        return imgs, classes


def get_dataset_constr(setting):
    if setting == 'unsupervised':
        return Unsupervised
    elif setting == 'supervised':
        return Supervised
    elif setting == 'semisup':
        return SemiSupervised
    elif setting == 'recons':
        return Reconstruction
    elif setting == 'superlat':
        return SupervisedLatents
    raise ValueError('Unrecognized setting "{}"'.format(setting))


def load_sprites(setting, data_filters=(None, None),
                 decision=None, train=True, path=None):

    train_filter, test_filter = data_filters

    if path is None:
        path = Dsprites.files['train']

    dataset_constr = get_dataset_constr(setting)
    if train:
        data = dataset_constr(*load_raw(path, train_filter),
                              target_transform=decision)
    else:
        data = dataset_constr(*load_raw(path, test_filter),
                              target_transform=decision)

    return data
