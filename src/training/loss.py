"""
Loss functions for training the models

The losses used to trained disentangled models are here. Most of the ones
analyzed in Locatello et al, 2020 are included, except DIP-VAE I and II.
Metrics for evaluation the models are also included.

Note: This might not be the best way to implement this. I have tried to keep
architectures, loss functions etc. as separated as possible. However, this
means that controlling the behaviour of some functions during training is
harder to achieve.
"""


import torch
import torch.nn as nn
import ignite.metrics as M
from torch.nn.functional import binary_cross_entropy_with_logits as logits_bce
from torch.nn.functional import mse_loss, cross_entropy
from torch.nn.modules.loss import _Loss

from .optimizer import init_optimizer


class AELoss(_Loss):
    """
    Base autoencoder loss
    """
    def __init__(self, reconstruction_loss='bce'):
        super().__init__(reduction='batchmean')
        if reconstruction_loss == 'bce':
            recons_loss = logits_bce
        elif reconstruction_loss == 'mse':
            recons_loss = mse_loss
        elif not callable(reconstruction_loss):
            raise ValueError('Unrecognized reconstruction'
                             'loss {}'.format(reconstruction_loss))

        self.recons_loss = recons_loss

    def forward(self, input, target):
        reconstruction, *latent_terms = input
        target = target.flatten(start_dim=1)

        recons_loss = self.recons_loss(reconstruction, target, reduction='sum')
        recons_loss /= target.size(0)

        latent_term = self.latent_term(*latent_terms)

        return recons_loss + latent_term

    def latent_term(self):
        raise NotImplementedError()


class GaussianVAELoss(AELoss):
    """
    This class implements the Variational Autoencoder loss with Multivariate
    Gaussian latent variables. With defualt parameters it is the one described
    in "Autoencoding Variational Bayes", Kingma & Welling (2014)
    [https://arxiv.org/abs/1312.6114].

    When $\beta>1$ this is the the loss described in $\beta$-VAE: Learning
    Basic Visual Concepts with a Constrained Variational Framework",
    Higgins et al., (2017) [https://openreview.net/forum?id=Sy2fzU9gl]
    """
    def __init__(self, reconstruction_loss='bce', beta=1.0,
                 beta_schedule=None):
        super().__init__(reconstruction_loss)
        self.beta = beta
        self.beta_schedule = beta_schedule
        self.anneal = 1.0

    def latent_term(self, z_sample, z_params):
        mu, logvar = z_params

        kl_div = _gaussian_kl(mu, logvar).sum()
        kl_div /= z_sample.size(0)
        return self.anneal * self.beta * kl_div

    def update_parameters(self, step):
        if self.beta_schedule is not None:
            steps, schedule_type, min_anneal = self.beta_schedule
            delta = 1 / steps

            if schedule_type == 'anneal':
                self.anneal = max(1.0 - step * delta, min_anneal)
            elif schedule_type == 'increase':
                self.anneal = min(min_anneal + delta * step, 1.0)


class CCIVAE(AELoss):
    """
    $\beta$-VAE trained with a constrained capacity increase loss (CCI-VAE).
    As in Burgess et al, 2018[https://arxiv.org/pdf/1804.03599.pdf%20].

    This loss slowly increases the strangth of the prior to force the models
    to "kill" unneccesary units in the latent representation.
    """
    def __init__(self, reconstruction_loss='bce', gamma=100.0,
                 capacity=0.0, capacity_schedule=None):
        super().__init__(reconstruction_loss)

        self.gamma = gamma
        self.capacity = capacity
        self.capacity_schedule = capacity_schedule

    def latent_term(self, z_sample, z_params):
        mu, logvar = z_params

        kl_div = _gaussian_kl(mu, logvar).sum()
        kl_div /= z_sample.size(0)
        return self.gamma * (kl_div - self.capacity).abs()

    def update_parameters(self, step):
        if self.capacity_schedule is not None:
            cmin, cmax, increase_steps = self.capacity_schedule
            delta = (cmax - cmin) / increase_steps

            self.capacity = min(cmin + delta * step, cmax)


class FactorLoss(AELoss):
    """
    FactorVAE loss as described in Disentangling by Factorizing,
    Kim & Mnih (2019) [https://arxiv.org/pdf/1802.05983.pdf].

    This loss uses adversarial training to minimize the total correlation
    while avoiding any penalization to the mutual information between input
    and latent codes (unlike $\beta-VAE).
    """
    def __init__(self, reconstruction_loss='bce',
                 gamma=10.0, gamma_schedule=None,
                 disc_args=None, optim_kwargs=None):
        super().__init__(reconstruction_loss)
        self.gamma = gamma
        self.gamma_schedule = gamma_schedule
        self.anneal = 1.0

        # if disc_args is None:
        #     disc_args = [('linear', [1000]), ('relu',)] * 6

        default_optim_kwargs = {'optimizer': 'adam', 'lr': 1e-5,
                                'betas': (0.5, 0.9)}
        if optim_kwargs is not None:
            default_optim_kwargs.update(optim_kwargs)

        # disc_args.append(('linear', [2]))
        # self.disc = feedforward.FeedForward(*disc_args)
        self.disc = nn.Sequential(
            nn.Linear(10, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 2),
        )

        self.optim = init_optimizer(params=self.disc.parameters(),
                                    **default_optim_kwargs)
        self._batch_samples = None

    @property
    def disc_device(self):
        return next(self.disc.parameters()).device

    def train(self, mode=True):
        self.disc.train(mode)
        for p in self.disc.parameters():
            p.requires_grad = mode

    def eval(self):
        self.train(False)

    def _set_device(self, input_device):
        if self.disc_device is None or (self.disc_device != input_device):
            self.disc.to(device=input_device)

    def latent_term(self, z_sample, z_params):
        mu, logvar = z_params

        # Hack to set the device
        self._set_device(mu.device)
        self.eval()

        z_sample1, z_sample2 = z_sample.chunk(2, 0)
        mu1, mu2 = mu.chunk(2, 0)
        logvar1, logvar2 = logvar.chunk(2, 0)

        kl_div = _gaussian_kl(mu1, logvar1).sum()
        kl_div /= z_sample1.size(0)

        log_z_ratio = self.disc(z_sample1)
        total_correlation = (log_z_ratio[:, 0] - log_z_ratio[:, 1]).mean()

        # print(total_correlation)
        self._batch_samples = z_sample1.detach(), z_sample2.detach()

        return kl_div + self.anneal * self.gamma * total_correlation

    def update_parameters(self, step):
        # update anneal value
        if self.gamma_schedule is not None:
            steps, min_anneal = self.gamma_schedule
            delta = 1 / steps
            self.anneal = max(1.0 - step * delta, min_anneal)

        if self._batch_samples is None:
            return

        # Train discriminator
        self.train()
        z1_samples, z2_samples = self._batch_samples
        self._batch_samples = None

        z_perm = _permute_dims(z2_samples)

        log_ratio_z = self.disc(z1_samples)
        log_ratio_z_perm = self.disc(z_perm)

        ones = torch.ones(z_perm.size(0), dtype=torch.long,
                          device=z_perm.device)
        zeros = torch.zeros_like(ones)

        disc_loss = 0.5 * (cross_entropy(log_ratio_z, zeros) +
                           cross_entropy(log_ratio_z_perm, ones))

        self.optim.zero_grad()
        disc_loss.backward()
        self.optim.step()


class ReconstructionNLL(_Loss):
    """
    Standard reconstruction of images. There are two options, minimize the
    Bernoulli loss (i.e. per pixel binary cross entropy) or MSE (i.e. Gaussian
    likelihoid).
    """
    def __init__(self, loss='bce'):
        super().__init__(reduction='batchmean')
        if loss == 'bce':
            recons_loss = logits_bce
        elif loss == 'mse':
            recons_loss = mse_loss
        elif not callable(loss):
            raise ValueError('Unrecognized reconstruction'
                             'loss {}'.format(loss))
        self.loss = recons_loss

    def forward(self, input, target):
        if isinstance(input, (tuple, list)):
            recons = input[0]
        else:
            recons = input

        return self.loss(recons, target, reduction='sum') / target.size(0)


class GaussianKLDivergence(_Loss):
    """
    Computes the KL divergence between a latent variable and a standard normal
    distribution. Optinally, allows for computing the KL for a single
    dimension. This can be used to see which units are being used by the model
    to solve the task.
    """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input, targets):
        _, _, (mu, logvar) = input

        if self.dim >= 0:
            mu, logvar = mu[:, self.dim], logvar[:, self.dim]

        kl = _gaussian_kl(mu, logvar).sum()
        return kl / targets.size(0)


def _permute_dims(latent_sample):
    pi = torch.randn_like(latent_sample).argsort(dim=0)
    perm = latent_sample[pi, range(latent_sample.size(1))]
    return perm


def _gaussian_kl(mean, logvar):
    return -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())


def get_loss(loss):
    loss_fn, params = loss['name'], loss['params']
    if loss_fn == 'vae':
        return GaussianVAELoss(**params, beta=1.0)
    elif loss_fn == 'beta-vae':
        return GaussianVAELoss(**params)
    elif loss_fn == 'factor-vae':
        return FactorLoss(**params)
    elif loss_fn == 'cci-vae':
        return CCIVAE(**params)
    elif loss_fn == 'recons_nll':
        return ReconstructionNLL(**params)
    elif loss_fn == 'bxent':
        return nn.BCEWithLogitsLoss(**params)
    elif loss_fn == 'xent':
        return nn.CrossEntropyLoss(**params)
    else:
        raise ValueError('Unknown loss function {}'.format(loss_fn))


def get_metric(metric):
    name = metric['name']
    params = metric['params']
    if name == 'mse':
        return M.MeanSquaredError(**params)
    elif name == 'vae':
        return M.Loss(GaussianVAELoss(**params))
    elif name == 'kl-div':
        return M.Loss(GaussianKLDivergence(**params))
    elif name == 'recons_nll':
        return M.Loss(ReconstructionNLL(**params))
    elif name == 'bxent':
        return M.Loss(nn.BCEWithLogitsLoss(**params))
    elif name == 'xent':
        return M.Loss(nn.CrossEntropyLoss(**params))
    elif name == 'acc':
        return M.Accuracy(**params)
    raise ValueError('Unrecognized metric {}.'.format(metric))


def init_metrics(loss, metrics):
    criterion = get_loss(loss)

    labels = [m.pop('label', m['name']) for m in metrics]
    metrics = {l: get_metric(m) for (l, m) in zip(labels, metrics)}

    return criterion, metrics
