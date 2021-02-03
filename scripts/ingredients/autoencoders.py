"""
Sacred Ingredient for Autoencoders

This ingredient takes an autoencoder configuration and produces an intance
of the specified generative model^*. it also has a convenience command used
to print the architecture.

This function will add the corresponding last layer to the model so that the
output of the encoder has the appropriate size. If the decoder layers are not
specified then it will try to transpose the definitions by itself^{**}.

To produce the models it uses the FeedForward module to stack the layers. This
is just a subclass of nn.Sequential with some added properties for convenience.
See 'src/models/feedforward.py' for more info.

^* Currently the only type fo model it supports is LGM
^{**} This might lead to an inconsistent definition where an unflatten layer is
applied **before** a ReLU activation instead of after. This has no functional
implications though. If you are feeling pedantic about these things you will
have to define your decoders explcitly.
"""


import sys
from sacred import Ingredient

if sys.path[0] != '../src':
    sys.path.insert(0, '../src')

from models.feedforward import FeedForward, transpose_layer_defs
from models.lgm import LGM


model = Ingredient('model')


# Print the model
@model.command(unobserved=True)
def show():
    model = init_lgm()
    print(model)


@model.capture
def init_lgm(gm_type, input_size, encoder_layers, latent_size,
             decoder_layers=None):

    encoder_layers += [('linear', [2 * latent_size])]
    encoder = FeedForward(input_size, encoder_layers, flatten=False)

    if decoder_layers is None:
        decoder_layers = encoder_layers[:-1]
        decoder_layers.append(('linear', [latent_size]))

        decoder_layers = transpose_layer_defs(decoder_layers, input_size)

    decoder = FeedForward(latent_size, decoder_layers, flatten=True)

    return LGM(latent_size, encoder, decoder)
