"""
Experiment script to train ground truth decoders the disentanglement datasets.

As with the standard end-to-end models, the decoders are trained to reconstruct
unseen images of novel combinations of generative factors. Contratry to the standard
models, the ground truth disentanglement is fed into the decoders as opposed to being
learned by them throught trianing.

The architectures defined are the same decoders used by the standard models, plus
some additions containing modifactions to depth and adding layers such as
batch normalization.

To run:
    cd <experiment-root>/scripts/
    python -m experiment.decoders with dataset.<option> model.<option> training.<option>

Additional configuration options can be achieved as explained in the Sacred documentation
[https://sacred.readthedocs.io/en/stable/]
"""

import sys
import numpy as np
import torch

from sacred import Experiment
from sacred.observers import FileStorageObserver
from ignite.engine import Events, create_supervised_evaluator, \
                          create_supervised_trainer

# Load experiment ingredients and their respective configs.
from ingredients.dataset import dataset, get_dataset, init_loader, \
                                get_data_spliters
from ingredients.decoders import model, create_decoder
from ingredients.training import training, init_metrics, init_optimizer, \
                                 reconstruction_loss, ModelCheckpoint

import configs.training as train_params
import configs.decoders as model_params

if '../src' not in sys.path:
    sys.path.append('../src')

from training.handlers import Tracer


# Set up experiment
ex = Experiment(name='decoders', ingredients=[dataset, model, training])

# Observers
ex.observers.append(FileStorageObserver.create('../data/sims/decoders'))
# ex.observers.append(MongoObserver.create(url='127.0.0.1:27017',
#                                          db_name='disent'))

# General configs
ex.add_package_dependency('torch', torch.__version__)
ex.add_config(no_cuda=False, save_folder='../data/sims/temp/dsprites')


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
dataset.add_config(dataset='sprites', setting='recons', shuffle=True)

dataset.add_named_config('dsprites', dataset='dsprites')
dataset.add_named_config('shapes3d', dataset='shapes3d', color_mode='rgb')
dataset.add_named_config('mpi3d', dataset='mpi3d', version='real')


# Training configs
training.config(train_params.generation)
training.add_config(loss=reconstruction_loss, metrics=[reconstruction_loss])

# Models
model.named_config(model_params.higgins)
model.named_config(model_params.burgess)
model.named_config(model_params.kim)
model.named_config(model_params.mpcnn)
model.named_config(model_params.kim_bn)
model.named_config(model_params.deep)
model.named_config(model_params.deep_bn)


# Experiment run
@ex.automain
def main(_config, save_folder):
    batch_size = _config['training']['batch_size']
    epochs = _config['training']['epochs']

    device = set_seed_and_device()

    init_dataset = get_dataset()
    data_filters = get_data_spliters()

    dataset = init_dataset(data_filters=data_filters, train=True)
    training_dataloader = init_loader(dataset, batch_size)

    # Init model
    n_gen_factors = training_dataloader.dataset.n_gen_factors
    img_size = training_dataloader.dataset.img_size
    model = create_decoder(latent_size=n_gen_factors,
                           img_size=img_size).to(device=device)

    # Init metrics
    loss, metrics = init_metrics()
    optimizer = init_optimizer(params=model.parameters())

    # Init engines
    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    validator = create_supervised_evaluator(model, metrics, device=device)

    # Exception for early termination
    @trainer.on(Events.EXCEPTION_RAISED)
    def terminate(engine, exception):
        if isinstance(exception, KeyboardInterrupt):
            engine.terminate()

    # Record training progression
    tracer = Tracer(metrics).attach(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training(engine):
        ex.log_scalar('training_loss', tracer.loss[-1])
        tracer.loss.clear()

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        validator.run(training_dataloader, epoch_length=100)

    @validator.on(Events.EPOCH_COMPLETED)
    def log_validation(engine):
        for metric, value in engine.state.metrics.items():
            ex.log_scalar('val_{}'.format(metric), value)

    # Attach model checkpoint
    def score_fn(engine):
        return -engine.state.metrics[list(metrics)[0]]

    checkpoint = ModelCheckpoint(
        dirname=_config['save_folder'],
        filename_prefix='decoder',
        score_function=score_fn,
        create_dir=True,
        require_empty=False,
        save_as_state_dict=True
    )
    validator.add_event_handler(Events.COMPLETED, checkpoint, {'model': model})

    # Run the training
    trainer.run(training_dataloader, max_epochs=epochs, epoch_length=10)
    # Select best model
    model.load_state_dict(checkpoint.last_checkpoint_state)

    # # Run on test data
    # test_dataloader = load_dataset(batch_size=batch_size,
    #                                data_filters=data_filters, train=False)

    # tester = create_supervised_evaluator(model, metrics, device=device)
    # test_metrics = tester.run(test_dataloader).metrics

    # # Save best model performance and state
    # for metric, value in test_metrics.items():
    #     ex.log_scalar('test_{}'.format(metric), value)

    ex.add_artifact(checkpoint.last_checkpoint, 'trained-model.pt')
