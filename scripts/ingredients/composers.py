"""
Sacred Ingredient for Composers

This ingredient takes a composer configuration and produces an intance
of the specified model^*. Like for the autoencoder models, it also has
a convenience command used to print the architecture.

The Composer Module is defined here as opposed to haveing it's own
source file. This was done mainly for the sake of convenience since
this is an experimental model, not a standard one as the LGM.

Different 'mixers' are defined here too. Only the results obtained
with the LinearMixer where added to the article, mainly because the
other ones offered no benefit when tested on the full dataset.

^* Currently the only type fo model it supports is LGM
"""


import sys
import torch
import torch.nn as nn
from sacred import Ingredient

if sys.path[0] != '../src':
    sys.path.insert(0, '../src')

from models.feedforward import FeedForward, transpose_layer_defs
from models.lgm import LGM

model = Ingredient('model')


# General model initialization function, init_fn will depend on the experiment
@model.capture
def init_model(init_fn, device='cpu'):
    model = init_fn()
    return model.to(device=device)


# Print the model
@model.command(unobserved=True)
def show():
    model = init_composer()
    print(model)


class LinearMixer(nn.Module):
    def __init__(self, n_actions, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.n_actions = n_actions
        self.ref_proj = nn.Linear(n_actions + latent_size, latent_size)
        self.trans_proj = nn.Linear(n_actions + latent_size, latent_size)

    def forward(self, z, actions):
        z_ref, z_trans = z.reshape(-1, 2, self.latent_size).chunk(2, 1)

        z_ref = torch.cat([z_ref.squeeze(1), actions], dim=1).contiguous()
        z_trans = torch.cat([z_trans.squeeze(1), actions], dim=1).contiguous()

        return self.ref_proj(z_ref) + self.trans_proj(z_trans)


class InterpolationMixer(nn.Module):
    def __init__(self, n_actions, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.n_actions = n_actions
        self.linear = nn.Linear(2 * latent_size + n_actions, latent_size)

    def forward(self, z, actions):
        z_ref, z_trans = z.reshape(-1, 2, self.latent_size).chunk(2, 1)
        z_ref, z_trans = z_ref.squeeze(1), z_trans.squeeze(1)

        w = self.linear(torch.cat([z_ref, z_trans, actions], dim=1)).sigmoid()
        return z_ref * w + z_trans * (1.0 - w)


class MLPMixer(nn.Module):
    def __init__(self, n_actions, latent_size):
        super().__init__()
        self.n_actions = n_actions
        self.latent_size = latent_size
        self.projection = FeedForward(2 * latent_size + n_actions,
                                      [('linear', [128]),
                                       ('relu',),
                                       ('batch_norm', [1]),
                                       ('linear', [latent_size])])

    def forward(self, z, actions):
        zpa = torch.cat([z.reshape(-1, self.latent_size), actions], dim=1)
        return self.projection(zpa.contiguous())


class Composer(nn.Module):
    def __init__(self, latent_size, n_actions, vae):
        super().__init__()
        self.vae = vae
        self.projector = LinearMixer(n_actions, latent_size)

    @property
    def latent_size(self):
        return self.vae.latent_size

    @property
    def n_actions(self):
        return self.projector.n_actions

    def forward(self, inputs):
        inputs, actions = inputs
        batch_size = inputs.size(0)

        # Format inputs so that we have shape (2 * batch_size, input_suze)
        # and corresponding reference and transform images follow each other
        inputs = inputs.flatten(0, 1).contiguous()

        # Compute latent values for the two images (reference and transform),
        # reshape the values so that corresponding pairs are in the same row
        z, params = self.vae.latent(self.vae.encoder(inputs))

        # Transform the latents according to the action and decode
        transformation = self.vae.decoder(self.projector(z, actions))

        return transformation, z.reshape(batch_size, 2, -1), params


@model.capture
def init_composer(gm_type, n_actions, input_size, encoder_layers, latent_size,
                  mixing_layer=None, decoder_layers=None):
    encoder_layers += [('linear', [2 * latent_size])]
    encoder = FeedForward(input_size, encoder_layers, flatten=False)

    if decoder_layers is None:
        decoder_layers = encoder_layers[:-1]
        decoder_layers.append(('linear', [latent_size]))

        decoder_layers = transpose_layer_defs(decoder_layers, input_size)

    decoder = FeedForward(latent_size, decoder_layers, flatten=True)

    lgm = LGM(latent_size, encoder, decoder)

    return Composer(latent_size, n_actions, lgm)
