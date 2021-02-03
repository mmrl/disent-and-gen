"""
Autoencoder model definitions

Following the Sacred configuration model using python functions, we define the
architectures of the models here. The names referece the first author of the
article from where they were take. Some might be slightly modified.

Configurations follow a general structure:
    1. gm_type (currently only lgm is available)
    2. latent_size
    3. input_size (not neccessary since it is overwritten depending on the dataset)
    4. encoder_layers: a list with layer definitions
    5. decoder layers: optional, model creation function will attemtp to transpose

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

def higgins():
    gm_type = 'lgm'
    latent_size = 10
    input_size = 4096

    encoder_layers = [
        ('linear', [1200]),
        ('relu',),
        ('linear', [1200]),
        ('relu',),
    ]

    decoder_layers = [
        ('linear', [1200]),
        ('tanh',),
        ('linear', [1200]),
        ('tanh',),
        ('linear', [1200]),
        ('tanh',),
        ('linear', [4096])
    ]


def burgess():
    gm_type = 'lgm'
    latent_size = 10
    input_size = 1, 64, 64

    encoder_layers = [
        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('flatten', [1]),

        ('linear', [256]),
        ('relu',),

        ('linear', [256]),
        ('relu',)
    ]


def kim():
    gm_type = 'lgm'
    latent_size = 10
    input_size = 3, 64, 64

    encoder_layers = [
        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('flatten', [1]),

        ('linear', [256]),
        ('relu',),
    ]


def burgess_v2():
    gm_type = 'lgm'
    latent_size = 10
    input_size = 1, 64, 64

    encoder_layers = [
        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('flatten', [1]),

        ('linear', [256]),
        ('relu',),

        ('linear', [256]),
        ('relu',)
    ]


def mathieu():
    gm_type = 'lgm'
    latent_size = 10
    input_size = 1, 64, 64

    encoder_layers = [
        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('flatten', [1]),

        ('linear', [128]),
        ('relu',),

        # ('linear', [512]),
        # ('relu',),
    ]

    decoder_layers = [
        # ('linear', [512]),
        # ('relu',),

        ('linear', [128]),
        ('relu',),

        ('linear', [4 * 4 * 64]),
        ('relu',),

        ('unflatten', (64, 4, 4)),

        # ('tconv', (64, 4, 2, 1)),
        # ('relu',),

        ('tconv', (64, 4, 2, 1)),
        ('relu',),

        ('tconv', (32, 4, 2, 1)),
        ('relu',),

        ('tconv', (32, 4, 2, 1)),
        ('relu',),

        ('tconv', (1, 4, 2, 1)),
    ]


# Similar as above but with max-pooling
def mpcnn():
    gm_type = 'lgm'
    latent_size = 10
    input_size = 1, 64, 64

    encoder_layers = [
        # n-channels, size, stride, padding
        ('conv', (32, 3, 1, 1)),
        # size, stride, padding, type
        ('pool', (2, 2, 0, 'max')),
        ('relu',),

        ('conv', (32, 3, 1, 1)),
        ('pool', (2, 2, 0, 'max')),
        ('relu',),

        ('conv', (32, 3, 1, 1)),
        ('pool', (2, 2, 0, 'max')),
        ('relu',),

        ('flatten', [1]),

        ('linear', [256]),
        ('relu',),

        ('linear', [256]),
        ('relu',),
    ]

    decoder_layers = [
        ('linear', [256]),
        ('relu',),

        ('linear', [256]),
        ('relu',),

        ('linear', [2048]),
        ('relu',),

        ('unflatten', (32, 8, 8)),

        ('upsample', (16, 16)),
        ('tconv', (32, 3, 1, 1)),
        ('relu',),

        ('upsample', (32, 32)),
        ('tconv', (32, 3, 1, 1)),
        ('relu',),

        ('upsample', (64, 64)),
        ('tconv', (32, 3, 1, 1)),
    ]
