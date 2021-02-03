"""
FeedForward module

This module initializes a list of layer configurations into a feed-forward
PyTorch module. This is class inherits from nn.Sequential and adds a few more
methods to build a sequential layer from a set of definitions.

Parameters in the config for each layer follow the order in Pytorch's
documentation. Excluding any of them will use the default ones. We can also
pass kwargs in a dict:

    ('layer_name', <list_of_args>, <dict_of_kwargs>)

This is a list of the configuration values supported:

Layer                   Paramaeters
==============================================================================
Convolution:            n-channels, size, stride, padding
Transposed Convolution: same, output_padding when stride > 1! (use kwargs)
Pooling:                size, stride, padding, type
Linear:                 output size, fit bias
Flatten:                start dim, (optional, defaults=-1) end dim
Unflatten:              unflatten shape (have to pass the full shape)
Batch-norm:             dimensionality (1-2-3d)
Upsample:               upsample_shape (hard to infer automatically). Bilinear
Non-linearity:          pass whatever arguments that non-linearity supports.

There is a method called transpose_layer_defs which allows for automatically
transposing the layer definitions for a decoder in a generative model. This
will automatically convert convolutions into transposed convolutions and
flattening to unflattening. However it will produce weird (but functionally
equivalent) orders of layers for ReLU before flattening, which means
unflattening in the corresponding decoder will be done before the ReLU.
"""


import math
import numpy as np
import torch
import torch.nn as nn


def _pair(s):
    if not isinstance(s, tuple):
        return s, s
    return s


def preprocess_defs(layer_defs):
    def preprocess(definition):
        if len(definition) == 1:
            return definition[0], [], {}
        elif len(definition) == 2 and isinstance(definition[1], (tuple, list)):
            return (*definition, {})
        elif len(definition) == 2 and isinstance(definition[1], dict):
            return definition[0], [], definition[1]
        elif len(definition) == 3:
            return definition
        raise ValueError('Invalid layer definition')

    return list(map(preprocess, layer_defs))


def get_nonlinearity(nonlinearity):
    if nonlinearity == 'relu':
        return nn.ReLU
    elif nonlinearity == 'sigmoid':
        return nn.Sigmoid
    elif nonlinearity == 'tanh':
        return nn.Tanh
    elif nonlinearity == 'lrelu':
        return nn.LeakyReLU
    elif nonlinearity == 'elu':
        return nn.ELU
    raise ValueError('Unrecognized non linearity: {}'.format(nonlinearity))


def create_linear(input_size, args, kwargs, transposed=False):
    if isinstance(input_size, (list, tuple)):
        in_features = input_size[-1]
    else:
        in_features = input_size

    if transposed:
        layer = nn.Linear(args[0], in_features, *args[1:], **kwargs)
    else:
        layer = nn.Linear(in_features, *args, **kwargs)

    if isinstance(input_size, (list, tuple)):
        input_size[-1] = args[0]
    else:
        input_size = args[0]

    return layer, input_size


def creat_batch_norm(ndims, input_size, args, kwargs):
    if ndims == 1:
        return nn.BatchNorm1d(input_size, *args, **kwargs)
    elif ndims == 2:
        return nn.BatchNorm2d(input_size[0], *args, **kwargs)
    elif ndims == 3:
        return nn.BatchNorm3d(input_size[0], *args, **kwargs)


def maxpool2d_out_shape(in_shape, pool_shape, stride, padding):
    in_channels, hout, wout = in_shape
    pool_shape = _pair(pool_shape)
    stride = _pair(stride)
    padding = _pair(padding)

    hval, wval = zip(pool_shape, stride, padding)

    hout = math.floor((hout - hval[0] + 2 * hval[2]) / hval[1]) + 1
    wout = math.floor((wout - wval[0] + 2 * wval[2]) / wval[1]) + 1

    return in_channels, hout, wout


def create_pool(kernel_size, stride, padding, mode, kwargs):
    if mode == 'avg':
        pooling = nn.AvgPool2d(kernel_size, stride, padding, **kwargs)
    elif mode == 'max':
        pooling = nn.MaxPool2d(kernel_size, stride, **kwargs)
    elif mode == 'adapt':
        pooling = nn.AdaptiveAvgPool2d(kernel_size, **kwargs)
    else:
        raise ValueError('Unrecognised pooling mode {}'.format(mode))

    return pooling


def conv2d_out_shape(in_shape, out_channels, kernel_shape, stride, padding):
    in_shape = in_shape[1:]
    kernel_shape = _pair(kernel_shape)
    stride = _pair(stride)
    padding = _pair(padding)

    hval, wval = zip(in_shape, kernel_shape, stride, padding)

    hout = math.floor((hval[0] - hval[1] + 2 * hval[3]) / hval[2]) + 1
    wout = math.floor((wval[0] - wval[1] + 2 * wval[3]) / wval[2]) + 1

    return out_channels, hout, wout


def transp_conv2d_out_shape(in_shape, out_channels, kernel_shape,
                            stride, padding):
    in_shape = in_shape[1:]
    kernel_shape = _pair(kernel_shape)
    stride = _pair(stride)
    padding = _pair(padding)

    hval, wval = zip(in_shape, kernel_shape, stride, padding)

    hout = (hval[0] - 1) * hval[2] - 2 * hval[3] + hval[1]
    wout = (wval[0] - 1) * wval[2] - 2 * wval[3] + wval[1]

    return out_channels, hout, wout


def compute_flattened_size(input_size, start_dim=1, end_dim=-1):
    start_dim -= 1
    if start_dim < 0:
        raise ValueError('Cannot flatten batch dimension')

    if end_dim < 0:
        end_dim = len(input_size) + 1

    output_size = list(input_size[:start_dim])
    output_size.append(np.prod(input_size[start_dim:end_dim]))
    output_size.extend(input_size[end_dim:])

    if len(output_size) == 1:
        return output_size[0]

    return output_size


class Unflatten(nn.Module):
    def __init__(self, unflatten_shape):
        super().__init__()
        self.unflatten_shape = unflatten_shape

    def forward(self, inputs):
        return inputs.view(-1, *self.unflatten_shape)

    def extra_repr(self):
        dims = [str(d) for d in self.unflatten_shape]
        return 'batch_size, {}'.format(', '.join(dims))


class FeedForward(nn.Sequential):
    def __init__(self, input_size, layer_defs, flatten=True):
        if isinstance(layer_defs, dict):
            layer_defs = dict.items()
        layer_defs = preprocess_defs(layer_defs)

        cnn_layers, output_size = [], input_size

        for layer_type, args, kwargs in layer_defs:
            if layer_type == 'linear':
                layer, output_size = create_linear(output_size, args, kwargs)
            elif layer_type == 'conv':
                layer = nn.Conv2d(output_size[0], *args, **kwargs)
                output_size = conv2d_out_shape(output_size, *args)
            elif layer_type == 'tconv':
                layer = nn.ConvTranspose2d(output_size[0], *args, **kwargs)
                output_size = transp_conv2d_out_shape(output_size, *args)
            elif layer_type == 'batch_norm':
                layer = creat_batch_norm(args[0], output_size,
                                         args[1:], kwargs)
            elif layer_type == 'pool':
                layer = create_pool(*args, kwargs)
                output_size = maxpool2d_out_shape(output_size, *args[:-1])
            elif layer_type == 'dropout':
                layer = nn.Dropout2d(*args, **kwargs)
            elif layer_type == 'flatten':
                layer = nn.Flatten(*args)
                output_size = compute_flattened_size(output_size)
            elif layer_type == 'unflatten':
                layer = Unflatten(args)
                output_size = args
            elif layer_type == 'upsample':
                layer = nn.UpsamplingBilinear2d(size=args)
                output_size = output_size[0], *args
            else:
                layer = get_nonlinearity(layer_type)(*args, **kwargs)

            cnn_layers.append(layer)

        super().__init__(*cnn_layers)

        self.input_size = input_size
        self.output_size = output_size
        self.flatten = flatten

    def forward(self, inputs):
        if isinstance(self.input_size, (list, tuple)):
            inputs = inputs.view(-1, *self.input_size)

        outputs = super().forward(inputs)

        if self.flatten:
            outputs = torch.flatten(outputs, start_dim=1)

        return outputs


def transpose_layer_defs(layer_defs, input_size):
    if isinstance(layer_defs, dict):
        layer_defs = layer_defs.items()

    layer_defs = preprocess_defs(layer_defs)

    transposed_layer_defs = []

    for layer_type, args, kwargs in layer_defs:
        if layer_type == 'linear':
            if isinstance(input_size, (tuple, list)):
                linear_size = *input_size[:-1], args[0]
                args = input_size[-1], *args[1:]
                input_size = linear_size
            else:
                args, input_size = [input_size] + args[1:], args[0]
        elif layer_type == 'conv':
            layer_type = 'tconv'
            conv_size = conv2d_out_shape(input_size, *args)
            args, input_size = (input_size[0], *args[1:]), conv_size
        elif layer_type == 'pool':
            layer_type = 'upsample'
            pooled_size = maxpool2d_out_shape(input_size, *args[:-1])
            args, input_size = input_size[1:], pooled_size
        elif layer_type == 'flatten':
            layer_type = 'unflatten'
            flattened_size = compute_flattened_size(input_size, *args)
            args, input_size = input_size, flattened_size

        layer = layer_type, args, kwargs
        transposed_layer_defs.append(layer)

    return list(reversed(transposed_layer_defs))
