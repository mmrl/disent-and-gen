"""
Sacred Ingredient for training functions.

The objective functions are defined and added as configureations to the
ingredient for ease of use. This allows chaging the objective function
easily and only needing to specify different parameters.
"""



import sys
from sacred import Ingredient
from ignite.engine import Events

if sys.path[0] != '../src':
    sys.path.insert(0, '../src')

from training.handlers import EarlyStopping, ModelCheckpoint, Tracer
from training.optimizer import init_optimizer, init_lr_scheduler
from training.loss import init_metrics as _init_metrics


def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = (y_pred.sigmoid() > 0.5).to(dtype=y.dtype)
    return y_pred, y


vae_loss = {'name': 'vae', 'params': {'reconstruction_loss': 'bce'}}
reconstruction_loss = {'name': 'recons_nll', 'params': {'loss': 'bce'}}
bvae_loss = {'name': 'beta-vae', 'params': {'reconstruction_loss': 'bce',
                                            'beta': 4.0}}
cap_const = {'name': 'constrained-beta-vae', 'params': {'reconstruction_loss':
                                                        'bce', 'gamma': 100,
                                                        'capacity': 7}}
bxent_loss = {'name': 'bxent', 'params': {}}
xent_loss = {'name': 'xent', 'params': {}}
accuracy = {'name': 'acc',
            'params': {'output_transform': thresholded_output_transform}}
mse_loss = {'name': 'mse', 'params': {}}
kl_div = {'name': 'kl-div', 'params': {}}


training = Ingredient('training')
training.add_named_config('vae', loss=vae_loss,
                          metrics=[reconstruction_loss, kl_div])
training.add_named_config('bvae', loss=bvae_loss,
                          metrics=[reconstruction_loss, kl_div])
training.add_named_config('capconst', loss=cap_const,
                          metrics=[reconstruction_loss, kl_div])
training.add_named_config('2afc', loss=bxent_loss,
                          metrics=[bxent_loss, accuracy])
training.add_named_config('mafc', loss=xent_loss,
                          metrics=[xent_loss, accuracy])
training.add_named_config('recons_nll', loss=reconstruction_loss,
                          metrics=[reconstruction_loss])


init_optimizer = training.capture(init_optimizer)


@training.capture
def init_metrics(loss, metrics):
    metrics = list(map(dict.copy, metrics))
    return _init_metrics(loss, metrics)
