"""
Experiment script to train standard autoencoder models on disentanglement datasets.

The models are trained on the standard reconstruction task. To test generalisation,
images corresponding to some generative factor combinations are excluded from the
training set. These are removed systematically to test different generalisation
settings.

The architectures and training objectives used are similar to the ones studied in
Locatello et al., 2019 (with the exception of DIP-VAE):
    1. VAE (Kingma & Welling,2014; Rezende & Mohammed 2014)
    2. $\beta$-VAE w/o CCI (Higgins et al, 2017; Burgess et al, 2018)
    3. FactorVAE (Kim & Mnih, 2019)

Note that to train a model on a dataset in this way, we need the generative factors
to be defined explicitly, otherwise we cannot exclude particular combinations.

To run:
    cd <experiment-root>/scripts/
    python -m experiment.disent with dataset.<option> model.<option> training.<option>

Additional configuration options can be achieved as explained in the Sacred documentation
[https://sacred.readthedocs.io/en/stable/]

"""


import sys
import numpy as np
import torch
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
from ignite.engine import Events, create_supervised_evaluator, \
                          create_supervised_trainer

# Load experiment ingredients and their respective configs.
from ingredients.dataset import dataset, get_dataset, init_loader, get_data_spliters
from ingredients.autoencoders import model, init_lgm
from ingredients.training import training, ModelCheckpoint, init_metrics, \
                                 init_optimizer
# from ingredients.analysis import compute_disentanglement

import configs.training as train_params
import configs.cnnvae as model_params

if '../src' not in sys.path:
    sys.path.append('../src')

from training.handlers import Tracer

# Set up experiment
ex = Experiment(name='disent', ingredients=[dataset, model, training])

# Observers
# ex.observers.append(FileStorageObserver.create('../data/sims/disent'))
ex.observers.append(MongoObserver.create(url='127.0.0.1:27017',
                                         db_name='disent'))

# General configs
ex.add_config(no_cuda=False, save_folder='../data/sims/dsprites')
ex.add_package_dependency('torch', torch.__version__)


# Functions

@ex.capture
def set_seed_and_device(seed, no_cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and not no_cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


# Dataset configs
dataset.add_config(dataset='sprites', setting='unsupervised', shuffle=True)

dataset.add_named_config('dsprites', dataset='dsprites')
dataset.add_named_config('shapes3d', dataset='shapes3d')
dataset.add_named_config('mpi3d', dataset='mpi3d', version='real')

# Training configs
training.config(train_params.vae)
training.named_config(train_params.beta)
training.named_config(train_params.cci)
training.named_config(train_params.factor)
training.named_config(train_params.bsched)
training.named_config(train_params.banneal)


# Model configs
model.named_config(model_params.higgins)
model.named_config(model_params.burgess)
model.named_config(model_params.burgess_v2)
model.named_config(model_params.mpcnn)
model.named_config(model_params.mathieu)
model.named_config(model_params.kim)


# Run experiment
@ex.automain
def main(_config, save_folder):
    batch_size = _config['training']['batch_size']
    epochs = _config['training']['epochs']

    device = set_seed_and_device()

    # Load data
    init_dataset = get_dataset()
    data_filters = get_data_spliters()

    dataset = init_dataset(data_filters=data_filters, train=True)
    training_dataloader = init_loader(dataset, batch_size)

    # Init model
    img_size = training_dataloader.dataset.img_size
    model = init_lgm(input_size=img_size).to(device=device)

    # Init metrics
    loss, metrics = init_metrics()
    optimizer = init_optimizer(params=model.parameters())

    # Init engines
    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    validator = create_supervised_evaluator(model, metrics, device=device)

    # Record training progression
    tracer = Tracer(metrics).attach(trainer)

    # Exception for early termination
    @trainer.on(Events.EXCEPTION_RAISED)
    def terminate(engine, exception):
        if isinstance(exception, KeyboardInterrupt):
            engine.terminate()

    @trainer.on(Events.ITERATION_COMPLETED)
    def update_loss_parameters(engine):
        loss.update_parameters(engine.state.iteration - 1)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training(engine):
        ex.log_scalar('training_loss', tracer.loss[-1])
        tracer.loss.clear()

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        validator.run(training_dataloader,
            epoch_length=np.ceil(len(training_dataloader) * 0.2))

    @validator.on(Events.EPOCH_COMPLETED)
    def log_validation(engine):
        for metric, value in engine.state.metrics.items():
            ex.log_scalar('val_{}'.format(metric), value)

    # Attach model checkpoint
    def score_fn(engine):
        return -engine.state.metrics[list(metrics)[0]]

    best_checkpoint = ModelCheckpoint(
        dirname=_config['save_folder'],
        filename_prefix='disent_best_nll',
        score_function=score_fn,
        create_dir=True,
        require_empty=False,
        save_as_state_dict=True
    )
    validator.add_event_handler(Events.COMPLETED, best_checkpoint,
                                {'model': model})

    # Save every 10 epochs
    periodic_checkpoint = ModelCheckpoint(
        dirname=_config['save_folder'],
        filename_prefix='disent_interval',
        n_saved=epochs//10,
        create_dir=True,
        require_empty=False,
        save_as_state_dict=True
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=10),
                              periodic_checkpoint, {'model': model})

    # Run the training
    trainer.run(training_dataloader, max_epochs=epochs)
    # Select best model
    model.load_state_dict(best_checkpoint.last_checkpoint_state)

    # # Run on test data
    # testing_dataloader = load_dataset(batch_size=batch_size,
    #                                   data_filters=data_filters, train=False)

    # tester = create_supervised_evaluator(model, metrics, device=device)
    # test_metrics = tester.run(testing_dataloader).metrics

    # # Save best model performance and state
    # for metric, value in test_metrics.items():
    #     ex.log_scalar('test_{}'.format(metric), value)

    ex.add_artifact(best_checkpoint.last_checkpoint, 'trained-model.pt')

    # Save all the periodic
    for name, path in periodic_checkpoint.all_paths:
        ex.add_artifact(path, name)
