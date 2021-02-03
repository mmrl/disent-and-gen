"""
Ground-truth decoder definitions

Following the Sacred configuration model using python functions, we define the
architectures of the models here. The names referece the generative model
architecture from where they are taken where appropriate.

Configurations follow a general structure:
    1. decoder layers: optional, model creation function will attemtp to transpose

Unlike the standard models we only need to specify the decoders.

Parameters in the config for each layer follow the order in Pytorch's documentation
Excluding any of them will use the default ones. We can also pass kwargs in a dict:

    ('layer_name', <list_of_args>, <dict_of_kwargs>)

This is a list of the configuration values supported:

Layer                   Paramaeters
==================================================================================
Convolution:            n-channels, size, stride, padding
Transposed Convolution: same, remeber output_padding when stride > 1! (use kwargs)
Pooling:                size, stride, padding, type
Linear:                 output size, fit bias
Flatten:                start dim, (optional, defaults=-1) end dim
Unflatten:              unflatten shape (have to pass the full shape)
Batch-norm:             dimensionality (1-2-3d)
Upsample:               upsample_shape (hard to infer automatically). Only bilinear
Non-linearity:          pass whatever arguments that non-linearity supports.
"""


# Simple MLP decoder
def higgins():
    decoder_layers = [
        ('linear', [1200]),
        ('tanh',),
        ('linear', [1200]),
        ('tanh',),
        ('linear', [1200]),
        ('tanh',),
        ('linear', [4096])
    ]


# Same decoder as in Burgess et al., 2018
def burgess():
    decoder_layers = [
        ('linear', [256]),
        ('relu',),

        ('linear', [4 * 4 * 32]),
        ('relu',),

        ('unflatten', (32, 4, 4)),

        ('tconv', (32, 4, 2, 1)),
        ('relu',),

        ('tconv', (32, 4, 2, 1)),
        ('relu',),

        ('tconv', (32, 4, 2, 1)),
        ('relu',),

        ('tconv', (-1, 4, 2, 1)) # use -1 to signal that conv must match target channels
    ]


def kim():
    decoder_layers = [
        ('linear', [256]),
        ('relu',),

        ('linear', [4 * 4 * 64]),
        ('relu',),

        ('unflatten', (64, 4, 4)),

        ('tconv', (64, 4, 2, 1)),
        ('relu',),

        ('tconv', (32, 4, 2, 1)),
        ('relu',),

        ('tconv', (32, 4, 2, 1)),
        ('relu',),

        ('tconv', (-1, 4, 2, 1)) # use -1 to signal that conv must match target channels
    ]


def kim_bn():
    decoder_layers = [
        ('linear', [256]),
        ('relu',),
        ('batch_norm', [1]),

        ('linear', [4 * 4 * 64, False]),
        ('relu',),
        ('batch_norm', [1]),

        ('unflatten', (64, 4, 4)),

        ('tconv', (64, 4, 2, 1), {'bias': False}),
        ('relu',),
        ('batch_norm', [2]),

        ('tconv', (64, 4, 2, 1), {'bias': False}),
        ('relu',),
        ('batch_norm', [2]),

        ('tconv', (32, 4, 2, 1), {'bias': False}),
        ('relu',),
        ('batch_norm', [2]),

        ('tconv', (-1, 4, 2, 1)) # use -1 to signal that conv must match target channels
    ]


def mpcnn():
    decoder_layers = [
        ('linear', [256]),
        ('relu',),
        ('batch_norm', [1]),

        ('linear', [256, False]),
        ('relu',),
        ('batch_norm', [1]),

        ('unflatten', (32, 3, 3)),

        ('tconv', (32, 3, 1, 1), {'bias': False}),
        ('relu',),
        ('batch_norm', [2]),

        ('tconv', (32, 3, 1, 1), {'bias': False}),
        ('relu',),
        ('batch_norm', [2]),

        ('tconv', (32, 3, 1, 1), {'bias': False}),
        ('relu',),
        ('batch_norm', [2]),

        ('tconv', (-1, 4, 2, 1)) # use -1 to signal that conv must match target channels
    ]


def deep():
    decoder_layers = [
        ('linear', [512]),
        ('relu',),

        ('linear', [512]),
        ('relu',),

        ('linear', [1024]),
        ('relu',),

        ('linear', [1024]),
        ('relu',),

        ('unflatten', (64, 4, 4)),

        ('tconv', (64, 4, 2, 1)),
        ('relu',),

        ('tconv', (64, 4, 2, 1)),
        ('relu',),

        ('tconv', (32, 4, 2, 1)),
        ('relu',),

        ('tconv', (-1, 4, 2, 1)) # use -1 to signal that conv must match target channels
    ]


def deep_bn():
    decoder_layers = [
        ('linear', [512]),
        ('relu',),
        ('batch_norm', [1]),

        ('linear', [512, False]),
        ('relu',),
        ('batch_norm', [1]),

        ('linear', [1024, False]),
        ('relu',),
        ('batch_norm', [1]),

        ('linear', [1024, False]),
        ('relu',),
        ('batch_norm', [1]),

        ('unflatten', (64, 4, 4)),

        ('tconv', (64, 4, 2, 1), {'bias': False}),
        ('relu',),
        ('batch_norm', [2]),

        ('tconv', (64, 4, 2, 1), {'bias': False}),
        ('relu',),
        ('batch_norm', [2]),

        ('tconv', (32, 4, 2, 1), {'bias': False}),
        ('relu',),
        ('batch_norm', [2]),

        ('tconv', (-1, 4, 2, 1), {'bias': False}) # use -1 to signal that conv must match target channels
    ]
