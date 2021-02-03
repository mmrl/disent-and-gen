"""
Sacred Ingredient for ground-truth decoders

This ingredient takes a decoder configuration and produces an intance
of the specified feedforward model^*. it also has a convenience command used
to print the architecture.

To produce the models it uses the FeedForward module to stack the layers. This
is just a subclass of nn.Sequential with some added properties for convenience.
See 'src/models/feedforward.py' for more info.
"""


import sys
import torch
import torch.nn as nn
from sacred import Ingredient

if sys.path[0] != '../src':
    sys.path.insert(0, '../src')

from models.feedforward import FeedForward

model = Ingredient('model')


# Print the model
@model.command(unobserved=True)
def show():
    model = init_decoder()
    print(model)


class GaussianNoise(nn.Module):
    def __init__(self, noise):
        super().__init__()
        # n_values = torch.as_tensor([1, 3, 6, 40, 32, 32])

        self.noise = noise
        # self.register_buffer('n_values', n_values)

    def forward(self, inputs, random_eval=False):
        if self.training or random_eval:
            eps = torch.randn_like(inputs)
            return inputs + self.noise * eps
        return inputs


# class GlobalStatsNorm(nn.Module):
#     def __init__(self):
#         super().__init__()
#         mean = torch.as_tensor([0.0, 2, 0.7, 180, 0.5, 0.5])
#         std = torch.as_tensor([1.0, 0.8, 0.17, 106.5, 0.3, 0.3])

#         self.register_buffer('mean', mean)
#         self.register_buffer('std', std)

#     def forward(self, inputs):
#         return (inputs - self.mean) / self.std


@model.capture
def init_decoder(latent_size, decoder_layers, img_size,
                 noisy_latents=True, learn_latent_stats=False):

    output_layer = decoder_layers[-1]
    output_layer_name, output_layer_args = output_layer[:2]

    n_channels = img_size[0]

    if output_layer_name == 'tconv':
        output_layer_args = [n_channels] + output_layer_args[1:]
    elif output_layer_name == 'linear':
        output_layer_args = ([n_channels * output_layer_args[0]] +
                             output_layer_args[1:])

    output_layer = [output_layer_name, output_layer_args] + output_layer[2:]

    decoder_layers = decoder_layers[:-1] + [output_layer]

    decoder = FeedForward(latent_size, decoder_layers)

    # noise_layer = GaussianNoise(0.01)

    # decoder_layers.insert(0, noise_layer)

    return decoder
