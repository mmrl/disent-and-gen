"""
Traversal of latent dimensions of generative models

Provides two functions, traverse_latent and traverse_all which give the
reconstructions and corresponding samples for a range of values along a
particlar or all the dimensions (one at a time).

Based on:
    https://github.com/YannDubs/disentangling-vae/blob/master/utils/visualize.py
"""

import numpy as np
import torch
from scipy import stats


def _get_traversal_range(mean=0, std=1, max_traversal=1.0):
    """Return the corresponding traversal range in absolute terms."""
    if max_traversal < 0.5:
        max_traversal = (1 - 2 * max_traversal) / 2  # from 0.45 to 0.05

        # from 0.05 to -1.645
        max_traversal = stats.norm.ppf(max_traversal, loc=mean, scale=std)

    # symmetrical traversals
    return (-1 * max_traversal, max_traversal)


def traverse_latent(model, idx, n_samples, data=None, max_traversal=1.0):
    model.train()
    device = next(model.parameters()).device

    if data is None:
        # mean of prior for other dimensions
        samples = torch.zeros(n_samples, model.latent_size)
        traversals = torch.linspace(*_get_traversal_range(max_traversal),
                                    steps=n_samples)

    else:
        if data.size(0) > 1:
            raise ValueError("Every value should be sampled from the same"
                             "posterior, but {} datapoints given.".format(
                                 data.size(0)))

        with torch.no_grad():
            h = model.encoder(data.to(device))
            samples, (post_mean, post_logvar) = model.latent(h)
            samples = samples.cpu().repeat(n_samples, 1)
            post_mean_idx = post_mean.cpu()[0, idx]
            post_std_idx = torch.exp(post_logvar / 2).cpu()[0, idx]

        # travers from the gaussian of the posterior in case quantile
        traversals = torch.linspace(*_get_traversal_range(mean=post_mean_idx,
                                                          std=post_std_idx),
                                    steps=n_samples)

    for i in range(n_samples):
        samples[i, idx] = traversals[i]

    recons = model.decoder(samples.to(device)).sigmoid().cpu().numpy()
    return recons, samples.cpu().numpy()


def traverse_all(model, n_samples, data=None):
    n_latents = model.latent_size

    recons, z_samples = [], []

    for i in range(n_latents):
        r, s = traverse_latent(model, i, n_samples, data)
        recons.append(r)
        z_samples.append(s)

    recons = np.stack(recons)
    z_samples = np.stack(z_samples)

    return recons, z_samples
