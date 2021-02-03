"""
PyTorch implementation of a Latent Gaussian Model (LGM).

This model implements an encoder-decoder pair with a latent space that learns
a generative model of the input distribution using Gaussian distributions. The
model exposes the standard forward method plus a few extrans to allow quick
sampling of the learned prior images and the posterior over latent variables
given an input. The usur can also get the mean embedding (discarding the
parameters using the LGM.embed function.

This is the model most commonly associated with VAEs. However we have opted
for the nomenclature in Rezende et. al, 2014 since this decouples the loss
from the architecture, which techincally can be used with non-variational
losses (as with the Wasserstein Autoencoder).
"""


import torch.nn as nn
from .stochastic import DiagonalGaussian, HomoscedasticGaussian
from .initialization import weights_init


def get_latent(latent_type):
    if latent_type == 'diagonal':
        return DiagonalGaussian
    elif latent_type == 'homoscedastic':
        return HomoscedasticGaussian
    raise ValueError('Unrecognized latent layer {}'.format(latent_type))


class LGM(nn.Module):
    def __init__(self, latent_size, encoder, decoder, latent_type='diagonal'):
        super().__init__()
        self.encoder = encoder
        self.latent = get_latent(latent_type)(latent_size)
        self.decoder = decoder

        self.reset_parameter()

    def reset_parameter(self):
        self.apply(weights_init)

    @property
    def nlayers(self):
        return len(self.encoder)

    @property
    def latent_size(self):
        return self.latent.size

    def posterior(self, inputs):
        """
        Return the posterior distribution given the inputs
        """
        h = self.encoder(inputs)
        return self.latent(h)[1]

    def decode(self, z):
        return self.decoder(z)

    def embed(self, inputs):
        """Embed a batch of data points, x, into their z representations."""
        h = self.encoder(inputs)
        return self.latent(h)[0]

    def forward(self, inputs):
        """
        Takes a batch of samples, encodes them, and then decodes them again.
        Returns the parameters of the posterior to enable ELBO computation.
        """
        h = self.encoder(inputs)
        z, z_params = self.latent(h)  # z_params = (mu, logvar)
        return self.decoder(z), z, z_params

    def sample(self, inputs=None, n_samples=1):
        """
        Sample from the prior distribution or the conditional posterior
        learned by the model. If no input is given the output will have
        size (n_samples, latent_size) and if mu and logvar are given it
        will have size (batch_size, n_samples, latent size)
        """
        zsamples = self.sample_latent(inputs, n_samples)
        return self.decode(zsamples)

    def sample_latent(self, inputs=None, n_samples=1):
        if inputs is None:
            h = self.encoder[-1].bias.new_zeros(1, 2 * self.latent_size)
        else:
            h = self.encoder(inputs)
        return self.latent.sample(h, n_samples)
