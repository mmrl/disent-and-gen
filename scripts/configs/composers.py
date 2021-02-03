"""
Transformers with VAEs for dsprites data

Parameters in the config follow the order in Pytorch's documentations
Excluding any of them will use the default ones
We can also pass kwargs w/o positional arguments

Convolution: n-channels, size, stride, padding
Transposed Convolution: same, remeber output_padding when stride > 1! (use kwargs)
Pooling: size, stride, padding, type
Linear: output size, fit bias
Flatten: start dim, (optional, defaults=-1) end dim
Unflatten: unflatten shape (have to pass the full shape)
Batch-norm: dimensionality (1-2-3d)
Upsample: upsample_shape (hard to infer automatically). Only bilinear is supported
Non-linearity: pass whatever arguments that non-linearity has
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
