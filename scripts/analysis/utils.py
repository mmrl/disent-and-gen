import os
import sys
from io import BytesIO
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

import json

from ingredients.autoencoders import init_lgm
from ingredients.decoders import init_decoder

import configs.datasplits as splits

if '../src' not in sys.path:
    sys.path.append('../src')

import dataset.shapes3d as shapes3d
import dataset.sprites as dsprites
import dataset.mpi as mpi3d

#################################### Load models ##############################

# loader = ExperimentLoader(
#     mongo_uri='127.0.0.1',
#     db_name='disent'
# )


# def load_vae_from_db(exp_id, device):
#     exp = loader.find_by_id(exp_id)

#     params = thaw(exp.config.model)
#     model = create_conv_vae(**params)

#     model_state = BytesIO(exp.artifacts['trained-model'].content)
#     model_state = torch.load(model_state)

#     model.load_state_dict(model_state)

#     return model.to(device=device).eval()


# def load_decoder_from_db(exp_id, device, img_size=None):
#     exp = loader.find_by_id(exp_id)

#     params = thaw(exp.config.model)

#     params['latent_size'] = 5

#     params.pop('transposed', None)
#     if img_size is not None:
#         params['output_size'] = img_size

#     model = create_decoder(**params)

#     model_state = BytesIO(exp.artifacts['trained-model'].content)
#     model_state = torch.load(model_state)

#     model.load_state_dict(model_state)

#     return model.to(device=device).eval()


def load_decoder(experiment_path, id, dataset, device):
    path = experiment_path + str(id) +'/'

    meta = path + 'config.json'
    param_vals = path + 'trained-model.pt'

    with open(meta) as f:
        architecture = json.load(f)['model']

    decoder = init_decoder(**architecture, img_size=dataset.img_size,
                           latent_size=dataset.n_factors)

    with open(param_vals, 'rb') as f:
        state_dict = torch.load(BytesIO(f.read()))

    decoder.load_state_dict(state_dict)

    return decoder.to(device=device).eval()


def load_lgm(experiment_path, id, device):
    path = experiment_path + str(id) + '/'

    meta = path + 'config.json'
    param_vals = path + 'trained-model.pt'

    with open(meta) as f:
        architecture = json.load(f)['model']

    lgm = init_lgm(**architecture)

    with open(param_vals, 'rb') as f:
        state_dict = torch.load(BytesIO(f.read()))

    lgm.load_state_dict(state_dict)

    return lgm.to(device=device).eval()


#################################### Load Data ################################

@dataclass
class DatasetWrapper:
    raw: tuple
    unsupervised_ctr: type
    supervised_ctr: type
    reconstruction_ctr: type

    def get_unsupervised(self):
        return self.unsupervised_ctr(*self.raw)

    def get_supervised(self):
        return self.supervised_ctr(*self.raw)

    def get_reconstruction(self):
        return self.reconstruction_ctr(*self.raw)

    @property
    def factors(self):
        return self.unsupervised_ctr.lat_names

    @property
    def n_factors(self):
        return self.unsupervised_ctr.n_gen_factors

    @property
    def img_size(self):
        return self.unsupervised_ctr.img_size


def partition_data(raw, mask):
    if len(raw) == 3:
        imgs, latents, latent_classes = raw
        idx = mask(latents, latent_classes)

        imgs = imgs[idx]
        latents = latents[idx]
        latent_classes = latent_classes[idx]

        return imgs, latents, latent_classes

    else:
        imgs, latents = raw
        idx = mask(latents)

        imgs = imgs[idx]
        latents = latents[idx]

        return imgs, latents


def load_dataset(dataset, condition, variant):
    if dataset == 'shapes3d':
        dataset_path = '../data/raw/shapes3d/3dshapes.h5'
        partition_masks = splits.Shapes3D.get_splits(condition, variant)
        dataset_module = shapes3d

    elif dataset == 'dsprites':
        dataset_path = '../data/raw/dsprites/dsprite_train.npz'
        partition_masks = splits.Dsprites.get_splits(condition, variant)
        dataset_module = dsprites

    elif dataset == 'mpi3d':
        dataset_path = '../data/raw/mpi/mpi3d_real.npz'
        partition_masks = splits.MPI3D.get_splits(condition, variant)
        dataset_module = mpi3d

    else:
        raise ValueError('Unrecognized dataset {}'.format(dataset))

    raw = dataset_module.load_raw(dataset_path)

    raw_train = partition_data(raw, partition_masks[0])
    raw_test = partition_data(raw, partition_masks[1])

    loaders = (dataset_module.Unsupervised,
               dataset_module.Supervised,
               dataset_module.Reconstruction)

    train_wrapper = DatasetWrapper(raw_train, *loaders)
    test_wrapper = DatasetWrapper(raw_test, *loaders)

    data = train_wrapper, test_wrapper

    return data


#============================== Save Data =====================================

def safe_save(data, path):
    if os.path.exists(path):
        old_data = np.load(path)
        for k in old_data:
            if k not in data:
                data[k] = old_data[k]

    np.savez(path, **data)
