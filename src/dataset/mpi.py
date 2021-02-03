"""
Disentanglement dataset from in Gondal et al 2019.

This dataset contains more realistic stimuli when compared to dSprites and
3Dshapes. Plus some combinations of factors have a tighther coupling between
them than others, which means that the models have an harder/easier time
learning how they interact.

For more info, the dataset can be found here:

arXiv preprint https://arxiv.org/abs/1906.03292
NeurIPS Challenge: https://www.aicrowd.com/challenges/
                           neurips-2019-disentanglement-challenge
"""
import numpy as np
import torch
import torchvision.transforms as trans
from itertools import product
from skimage.color import rgb2hsv
from torch.utils.data import Dataset, DataLoader
# from functools import partialmethod


def load_raw(path, latent_filter=None):
    data_zip = np.load(path, allow_pickle=True)
    images = data_zip['images']

    latent_values = list(product(*MPI3D.lat_values.values()))
    latent_values = np.asarray(latent_values, dtype=np.int8)

    if latent_filter is not None:
        idx = latent_filter(latent_values)

        images = images[idx]
        latent_values = latent_values[idx]

    return images, latent_values


class MPI3D(Dataset):
    """
    #==========================================================================
    # Latent Dimension,    Latent values                                 N vals
    #==========================================================================

    # object color:        white=0, green=1, red=2, blue=3,                  6
    #                      brown=4, olive=5
    # object shape:        cone=0, cube=1, cylinder=2,                       6
    #                      hexagonal=3, pyramid=4, sphere=5
    # object size:         small=0, large=1                                  2
    # camera height:       top=0, center=1, bottom=2                         3
    # background color:    purple=0, sea green=1, salmon=2                   3
    # horizontal axis:     40 values liearly spaced [0, 39]                 40
    # vertical axis:       40 values liearly spaced [0, 39]                 40
    """
    files = {"toy": "../data/raw/mpi/mpi3d_toy.npz",
             "realistic": "../data/raw/mpi/mpi3d_realistic.npz",
             "real": "../data/raw/mpi/mpi3d_real.npz"}

    n_gen_factors = 7
    lat_names = ('object_color', 'object_shape', 'object_size', 'camera_height',
                 'background_color', 'horizontal_axis', 'vertical_axis')
    lat_sizes = np.array([6, 6, 2, 3, 3, 40, 40])

    img_size = (3, 64, 64)

    lat_values = {'object_color': np.arange(6),
                  'object_shape': np.arange(6),
                  'object_size': np.arange(2),
                  'camera_height': np.arange(3),
                  'background_color': np.arange(3),
                  'horizontal_axis': np.arange(40),
                  'vertical_axis': np.arange(40)}

    def __init__(self, imgs, latent_values, color_mode='rgb'):
        self.imgs = imgs
        self.latent_values = latent_values

        image_transforms = [trans.ToTensor(),
                            trans.ConvertImageDtype(torch.float32),
                            trans.Lambda(lambda x: x.flatten())]

        if color_mode == 'hsv':
            image_transforms.insert(0, trans.Lambda(rgb2hsv))

        latent_transforms = [trans.Lambda(lambda x: x * self.lat_sizes),
                             trans.Lambda(lambda x: torch.from_numpy(x).to(
                                 dtype=torch.float32))]

        self.transform = trans.Compose(image_transforms)
        self.target_transform = trans.Compose(latent_transforms)

    def __getitem__(self, key):
        return (self.imgs[key],
                self.latent_values[key])

    def __len__(self):
        return len(self.imgs)


class Unsupervised(MPI3D):
    def __getitem__(self, key):
        imgs = self.transform(self.imgs[key])
        return imgs, imgs


class Supervised(MPI3D):
    def __getitem__(self, idx):
        imgs, latent_values = self.imgs[idx], self.latent_values[idx]

        imgs = self.transform(imgs)
        classes = self.target_transform(latent_values)

        return imgs, classes


class Reconstruction(MPI3D):
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
    # elif setting == 'semi-supervised':
    #     return SemiSupervisedLoader
    elif setting == 'recons':
        return Reconstruction
    # elif setting == 'persup':
    #     return SupervisedFromTrueLatents
    raise ValueError('Unrecognized setting "{}"'.format(setting))


def load_mpi3d(setting, version='real', data_filters=(None, None),
               decision=None, color_mode='rgb', train=True, path=None):

    if version not in ['toy', 'realistic', 'real']:
        raise ValueError('Unrecognized datset version {}'.format(version))

    if path is None:
        path = MPI3D.files[version]

    dataset_constr = get_dataset_constr(setting)
    if train:
        data = dataset_constr(*load_raw(path, data_filters[0]),
                              color_mode=color_mode)
    else:
        data = dataset_constr(*load_raw(path, data_filters[1]),
                              color_mode=color_mode)

    return data
