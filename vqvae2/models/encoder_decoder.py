
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vqvae2.models.residual import ResidualStack

imsize48_default_architecture = dict(
    conv_args=dict(
        kernel_sizes=[5, 3, 3],
        n_channels=[16, 32, 64],
        strides=[3, 2, 2],
        res_h_dim=32,
        n_res_layers=3,
    ),
    conv_kwargs=dict(
        hidden_sizes=[],
        batch_norm_conv=False,


    ),
    deconv_args=dict(
        hidden_sizes=[],

        deconv_input_width=3,
        deconv_input_height=3,
        deconv_input_channels=64,

        deconv_output_kernel_size=6,
        deconv_output_strides=3,
        deconv_output_channels=3,

        kernel_sizes=[3, 3],
        n_channels=[32, 16],
        strides=[2, 2],
        res_h_dim=32,
        n_res_layers=3,
    ),
    deconv_kwargs=dict(
        batch_norm_deconv=False,
        batch_norm_fc=False,

    )
)


def identity(x):
    return x


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """
    """
        conv_args, conv_kwargs, deconv_args, deconv_kwargs = \
            architecture['conv_args'], architecture['conv_kwargs'], \
            architecture['deconv_args'], architecture['deconv_kwargs']
        conv_output_size = deconv_args['deconv_input_width'] * \
            deconv_args['deconv_input_height'] * \
            deconv_args['deconv_input_channels']

        self.encoder = encoder_class(
            **conv_args,
            paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
            input_height=self.imsize,
            input_width=self.imsize,
            input_channels=self.input_channels,
            output_size=conv_output_size,
            init_w=init_w,
            hidden_init=hidden_init,
            **conv_kwargs)
    """

    def __init__(self,
                 input_width,
                 input_height,
                 input_channels,
                 output_size,
                 kernel_sizes,
                 n_channels,
                 strides,
                 res_h_dim,
                 n_res_layers,
                 paddings,
                 hidden_sizes=None,
                 batch_norm_conv=False,
                 init_w=1e-4,
                 hidden_init=nn.init.xavier_uniform_,
                 hidden_activation=nn.ReLU(),
                 output_activation=identity,
                 verbose=False,
                 **kwargs
                 ):
        super(Encoder, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.batch_norm_conv = batch_norm_conv

        self.conv_input_length = self.input_width * \
            self.input_height * self.input_channels

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.res_h_dim = res_h_dim
        self.n_res_layers = n_res_layers

        for out_channels, kernel_size, stride, padding in \
                zip(n_channels, kernel_sizes, strides, paddings):
            conv = nn.Conv2d(input_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding)
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels

        # find output dim of conv_layers by trial and add normalization conv layers
        test_mat = torch.zeros(1, self.input_channels, self.input_width,
                               self.input_height)  # initially the model is on CPU (caller should then move it to GPU if
        if verbose:
            print(test_mat.shape)
        for conv_layer in self.conv_layers:
            test_mat = conv_layer(test_mat)
            if verbose:
                print(test_mat.shape)
            self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))
            h_dim = test_mat.shape[1]
            res_h_dim = 32
            n_res_layers = 3
        self.conv_layers.append(ResidualStack(
            h_dim, h_dim, res_h_dim, n_res_layers)
        )

    def forward(self, x):

        for conv, norm in zip(self.conv_layers, self.conv_norm_layers):
            x = conv(x)
            x = norm(x)
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            hidden_sizes,

            deconv_input_width,
            deconv_input_height,
            deconv_input_channels,

            deconv_output_kernel_size,
            deconv_output_strides,
            deconv_output_channels,

            kernel_sizes,
            n_channels,
            strides,
            res_h_dim,
            n_res_layers,

            paddings,

            batch_norm_deconv=False,
            batch_norm_fc=False,
            init_w=1e-3,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
            verbose=False,
            **kwargs
    ):
        assert len(kernel_sizes) == \
            len(n_channels) == \
            len(strides) == \
            len(paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation

        self.deconv_input_width = deconv_input_width
        self.deconv_input_height = deconv_input_height
        self.deconv_input_channels = deconv_input_channels
        deconv_input_size = self.deconv_input_channels * \
            self.deconv_input_height * self.deconv_input_width
        self.batch_norm_deconv = batch_norm_deconv

        self.deconv_layers = nn.ModuleList()
        self.deconv_norm_layers = nn.ModuleList()

        self.res_h_dim = res_h_dim
        self.n_res_layers = n_res_layers

        for n, (out_channels, kernel_size, stride, padding) in \
                enumerate(zip(n_channels, kernel_sizes, strides, paddings)):
            deconv = nn.ConvTranspose2d(deconv_input_channels,
                                        out_channels,
                                        kernel_size,
                                        stride=stride,
                                        padding=padding)
            hidden_init(deconv.weight)
            deconv.bias.data.fill_(0)

            deconv_layer = deconv
            self.deconv_layers.append(deconv_layer)
            deconv_input_channels = out_channels
            if n == 0:
                self.deconv_layers.append(ResidualStack(
                    out_channels, out_channels, self.res_h_dim, self.n_res_layers),)

        test_mat = torch.zeros(1, self.deconv_input_channels,
                               self.deconv_input_width,
                               self.deconv_input_height)  # initially the model is on CPU (caller should then move it to GPU if

        if verbose:
            print(test_mat.shape)
        for deconv_layer in self.deconv_layers:
            test_mat = deconv_layer(test_mat)
            if verbose:
                print(test_mat.shape)
            self.deconv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))

        self.first_deconv_output = nn.ConvTranspose2d(
            deconv_input_channels,
            deconv_output_channels,
            deconv_output_kernel_size,
            stride=deconv_output_strides,
        )
        hidden_init(self.first_deconv_output.weight)
        self.first_deconv_output.bias.data.fill_(0)

    def forward(self, input):

        h = self.apply_forward(input, self.deconv_layers, self.deconv_norm_layers,
                               use_batch_norm=self.batch_norm_deconv)
        first_output = self.output_activation(self.first_deconv_output(h))
        return first_output

    def apply_forward(self, input, hidden_layers, norm_layers,
                      use_batch_norm=False):
        h = input
        for layer, norm_layer in zip(hidden_layers, norm_layers):
            h = layer(h)
            if use_batch_norm:
                h = norm_layer(h)
            h = self.hidden_activation(h)
        return h


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((16, 3, 48, 48))
    x = torch.tensor(x).float()

    # test encoder
    #encoder = Encoder(40, 128, 3, 64)
    #encoder_out = encoder(x)
    #print('Encoder out shape:', encoder_out.shape)
    architecture = imsize48_default_architecture
    conv_args, conv_kwargs, deconv_args, deconv_kwargs = \
        architecture['conv_args'], architecture['conv_kwargs'], \
        architecture['deconv_args'], architecture['deconv_kwargs']
    conv_output_size = deconv_args['deconv_input_width'] * \
        deconv_args['deconv_input_height'] * \
        deconv_args['deconv_input_channels']
    imsize = 48
    encoder = Encoder(
        **conv_args,
        paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
        input_height=imsize,
        input_width=imsize,
        input_channels=3,
        output_size=conv_output_size,
        init_w=1e-4,
        hidden_init=nn.init.xavier_uniform_,
        verbose=True,
        **conv_kwargs)

    z = encoder(x)
    print(z.shape, z.mean(), z.std())

    decoder = Decoder(
        **deconv_args,
        init_w=1e-3,
        output_activation=identity,
        paddings=np.zeros(
            len(deconv_args['kernel_sizes']), dtype=np.int64),
        hidden_init=nn.init.xavier_uniform_,
        verbose=True,
        **deconv_kwargs)

    x = decoder(z)
    print(x.shape, x.mean(), x.std())
