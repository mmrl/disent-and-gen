"""
Analsys script for disentangled models

This script takes a dataset, conditon and variant and computes the differnet
metrics for the specified models. These are defined in the conditons.yaml file

The metrics computed include traning and validation reconstruction scores and
the disentanglement metrics defined by the DCI framewwork (Eastwood &
Williams, 2019). These are plotted and saved to the specified folder.
"""


import sys
import os
import argparse
import yaml
from enum import Enum

import numpy as np
import torch
from torch.utils.data import DataLoader

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import load_dataset, load_decoder, load_lgm, safe_save

if '../src' not in sys.path:
    sys.path.append('../src')

from training.loss import get_metric

from analysis.metrics import compute_dci_metrics, dci2df
from analysis.hinton import plot_hinton_matrices
from analysis.testing import get_recons, model_scores

# Disable grad globally
torch.set_grad_enabled(False)

# Config options
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

sns.set(color_codes=True)
sns.set_style("white", {'axes.grid': False})


def generate_recons(data, end2end_models, gt_decoder, model_names,
                    img_size, n_examples):

    loader = DataLoader(data, n_examples, shuffle=True)
    latents, inputs = next(iter(loader))

    recons = [get_recons(inputs.flatten(start_dim=1), vae, device)
              for vae in end2end_models]
    recons = recons + [get_recons(latents, gt_decoder, device)]

    return np.stack([inputs.numpy()] + recons)


def plot_reconstructions(all_images, model_names, img_size,
                         save_file, fig=None):
    n_examples = all_images.shape[1]
    if fig is None:
        fig, axes = plt.subplots(n_examples, len(model_names) + 2,
                                 figsize=(20, 10))
    else:
        axes = fig.axes

    labels = ['original'] + model_names + ['GT Decoder']

    for i, instance_recons in enumerate(all_images.transpose(1, 0, 2)):
        for j, (img, lab) in enumerate(zip(instance_recons, labels)):
            img = img.reshape(*img_size).transpose(1, 2, 0).squeeze()

            if img_size[0] == 1:
                axes[i, j].imshow(img, vmin=0, vmax=1, cmap='Greys_r')
            else:
                axes[i, j].imshow(img, vmin=0, vmax=1)

            if i == 0:
                axes[i, j].set_title(lab)

    for ax in axes.reshape(-1):
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    fig.savefig(save_file)


def plot_disentanglement_scores(overall_disent, disentanglement, completeness,
                                save_file):
    sns.set_style("white", {'axes.grid': True})
    disent_fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5),
                                               sharey=True)

    sns.barplot(data=disentanglement.reset_index(), hue='models',
                x='latent', y='disentanglement', ax=ax1).set(ylim=(0, 1.03))
    sns.barplot(data=overall_disent.reset_index(), hue='models',
                x='models', y='overall disentanglement',
                ax=ax2).set(ylim=(0, 1.03))
    sns.barplot(data=completeness.reset_index(), hue='models',
                x='factor', y='completeness', ax=ax3).set(ylim=(0, 1.03))

    ax2.tick_params(axis='x', labelrotation=30)
    ax3.tick_params(axis='x', labelrotation=30)

    for ax in [ax1, ax2, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax1.legend_.remove()
    ax2.legend_.remove()
    # ax3.legend_.remove()

    return disent_fig.savefig(save_file)


def plot_disent_vs_nll(all_metrics, save_file):
    sns.set_style("white", {'axes.grid': False})

    f = sns.relplot(kind='scatter', data=all_metrics,
                    x='Disentanglement', y='NLL', hue='Models',
                    style='Data', s=150, legend=True)

    for ax in f.axes.ravel():
        ax.set(xlim=(0, 1.03))

    # f.axes[0, 0].set(xlim=(0, 1.05), ylim=(3400, 22500), yscale='log')

    for name, g in all_metrics.groupby(['Models'], sort=False):
        y, x = g['NLL'].values, g['Disentanglement'].values
        plt.plot(x, y)

    # plt.fill_between([0, 1.02], 3400, 3600, color = 'k', alpha = 0.1)

    f.savefig(save_file)


def disentanglement_vs_nll(raw_train, raw_test, end2end_models, gt_decoder,
                           model_names, overall_disent):
    metric = {'nll': get_metric({'name': 'recons_nll', 'params': {}})}

    train = raw_train.get_unsupervised()
    train_nll = model_scores(end2end_models, train, model_names,
                             metric, device)

    train = raw_train.get_reconstruction()
    decoder_train_nll = model_scores([gt_decoder], train, ['GT Decoder'],
                                     metric, device)

    test = raw_test.get_unsupervised()
    test_nll = model_scores(end2end_models, test, model_names, metric, device)

    test = raw_test.get_reconstruction()
    decoder_test_nll = model_scores([gt_decoder], test, ['GT Decoder'],
                                    metric, device)

    all_train_nll = pd.concat([train_nll, decoder_train_nll])
    all_test_nll = pd.concat([test_nll, decoder_test_nll])

    score_names = ['Training', 'Test']

    recons_metrics = pd.concat([all_train_nll, all_test_nll], keys=score_names,
                               names=['Data', 'Models'])
    recons_metrics.name = 'NLL'

    disent = np.concatenate([overall_disent.values, np.asarray([1.0])])
    disent = np.repeat(disent.reshape(1, -1), 2, axis=0).ravel()

    all_metrics = recons_metrics.reset_index()
    all_metrics.insert(3, 'Disentanglement', disent)

    return all_metrics


def run_analysis(end2end_models, gt_decoder, dataset,
                 model_names, save_folder=''):
    raw_train, raw_test = dataset
    factors = raw_train.factors
    factors = list(map(lambda s: s.replace('_', ' '), factors))

    # Plot example reconstructions for training data

    train = raw_train.get_reconstruction()
    train_images = generate_recons(train, end2end_models, gt_decoder,
                                   model_names, train.img_size, 5)

    plot_reconstructions(train_images, model_names, train.img_size,
                         save_file=save_folder + 'train-recons.pdf')

    # Plot example reconstructions for test data

    test = raw_test.get_reconstruction()
    test_images = generate_recons(test, end2end_models, gt_decoder,
                                  model_names, train.img_size, 5)

    plot_reconstructions(test_images, model_names, train.img_size,
                         save_file=save_folder + 'test-recons.pdf')

    # Compute disentanglment metrics
    dci_results = compute_dci_metrics(end2end_models,
                                      raw_train.get_supervised())

    R_matrices = [r.R_coeff for r in dci_results]
    overall_disent, disentanglement, completeness = dci2df(dci_results,
                                                           model_names,
                                                           factors)

    # Plot Hinton matrices
    hinton_fig = plot_hinton_matrices(R_matrices, model_names, factors)
    hinton_fig.savefig(save_folder + 'hinton.pdf')

    # Plot disentanglment scores
    plot_disentanglement_scores(overall_disent, disentanglement, completeness,
                                save_folder + 'disent-scores.pdf')

    # Disentanglement to Reconstruction plot
    all_metrics = disentanglement_vs_nll(raw_train, raw_test, end2end_models,
                                         gt_decoder, model_names,
                                         overall_disent)

    plot_disent_vs_nll(all_metrics, save_folder + 'disent_vs_nll.pdf')

    analysis_data = {
        'model_names': model_names + ['GT Decoder'],
        'train_recons': train_images,
        'test_recons': test_images,
        'metrics': all_metrics,
        'R_coeff': np.stack(R_matrices)
    }

    return analysis_data


def main(dataset, condition, variant, sims_ids, model_names,
         gt_dec_id, sims_root):
    plots_folder = '../plots/disent/{}/{}/{}/'.format(dataset, condition,
                                                      variant)
    results_folder = '../data/results/disent/{}/{}/'.format(dataset, condition)

    print('Analysing dataset: {}\n\t  condition: {}\n\t  variant: {}'
          '\nLoading dataset...'.format(dataset, condition, variant))

    dataset = load_dataset(dataset, condition, variant)

    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    print('Done.\nLoading models...')

    end2end_models = [load_lgm(sims_root, _id, device) for _id in sims_ids]
    decoder = load_decoder(sims_root.replace('disent', 'decoders'),
                           gt_dec_id, dataset[0], device)

    print('Done.\nRunning analysis...')

    analysis_data = run_analysis(end2end_models, decoder, dataset,
                                 model_names, plots_folder)

    analysis_data['run_ids'] = sims_ids
    analysis_data['gt_dec_id'] = gt_dec_id

    if 'metrics' in analysis_data:
        metrics = analysis_data.pop('metrics')
        metrics.to_csv(results_folder + '{}-metrics.csv'.format(variant))
        safe_save(analysis_data,
                  results_folder + '{}-images.npz'.format(variant))

    print('Finished. Results have been saved to: \"{}\"\n'
          'Plots have been saved to \"{}\"'.format(results_folder,
                                                   plots_folder))


parser = argparse.ArgumentParser(
        description='Anlysis script for disentangled models')


class Datasets(Enum):
    DSPRITES = 'dsprites'
    SHAPES3D = 'shapes3d'
    MPI3D = 'mpi3d'


class Conditions(Enum):
    R2E = 'r2e'
    R2R = 'r2r'
    EXTRP = 'extrp'


parser.add_argument('-d', type=Datasets, required=True, dest='dataset',
                    choices=Datasets, help='Dataset on which to run the analysis.'
                                           'One of [dsprites, shapes3d].')
parser.add_argument('-c', type=Conditions, required=True, dest='condition',
                    choices=Conditions, help='The generalisation settings.'
                                             'One of [r2e, r2r, extrp')
parser.add_argument('-v', type=str, required=True, dest='variant',
                    help='The different variants for each condition')

if __name__ == "__main__":
    args = parser.parse_args()

    with open('analysis/conditions.yaml') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    analysis_args = configs[args.dataset.value][args.condition.value][args.variant]

    main(**analysis_args)
