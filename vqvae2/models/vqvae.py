
import torch
import torch.nn as nn
import numpy as np
from vqvae2.models.encoder_decoder import Encoder, Decoder
from vqvae2.models.quantizer import VectorQuantizer
from vqvae2.pixelcnn.models import GatedPixelCNN
from vqvae2.config import imsize48_default_architecture


def identity(x):
    return x


class VQVAE(nn.Module):
    def __init__(self, architecture, imsize,
                 n_embeddings, embedding_dim, beta, gpu_id=0, input_channels=3, verbose=False, with_pixelcnn=True):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
   
        self.imsize = imsize
        self.input_channels = input_channels
        self.imlength = self.imsize * self.imsize * self.input_channels
        conv_args, conv_kwargs, deconv_args, deconv_kwargs = \
            architecture['conv_args'], architecture['conv_kwargs'], \
            architecture['deconv_args'], architecture['deconv_kwargs']
        conv_output_size = deconv_args['deconv_input_width'] * \
            deconv_args['deconv_input_height'] * \
            deconv_args['deconv_input_channels']

        self.encoder = Encoder(
            **conv_args,
            paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
            input_height=imsize,
            input_width=imsize,
            input_channels=input_channels,
            output_size=conv_output_size,
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            verbose=verbose,
            **conv_kwargs)  # .to(self.device)

        h_dim = conv_args['n_channels'][-1]
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)  # .to(self.device)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta, gpu_id=gpu_id)
        # decode the discrete latent representation

        self.pre_dequantization_conv = nn.ConvTranspose2d(embedding_dim,
                                                          n_embeddings,
                                                          kernel_size=1,
                                                          stride=1)

        self.decoder = Decoder(
            **deconv_args,
            init_w=1e-3,
            output_activation=identity,
            paddings=np.zeros(
                len(deconv_args['kernel_sizes']), dtype=np.int64),
            hidden_init=nn.init.xavier_uniform_,
            verbose=verbose,
            **deconv_kwargs)  # .to(self.device)

        """
        Initialize PixelCNN for autoregressive sampling
        """
        # get latent image dim
        test_mat = torch.zeros(1, input_channels, imsize, imsize)
        test_z = self.encoder(test_mat)
        self.z_dim = test_z.shape[-1]
        self.embedding_dim = embedding_dim
        self.representation_size = self.z_dim**2*embedding_dim
        if with_pixelcnn:
            pixelcnn_kwargs = architecture['pixelcnn_kwargs']

            self.pixelcnn = GatedPixelCNN(
                n_embeddings, self.z_dim ** 2, pixelcnn_kwargs['n_layers'])

    def forward(self, x, verbose=False, include_binaries=False):

        if len(x.shape) != 4:
            x = x.view(x.shape[0], self.imsize,
                       self.imsize, self.input_channels)
            x = x.permute(0, 3, 1, 2)

        z_e = self.encoder(x)
        if verbose:
            print('In  ', x.shape)
            print("z_e1", z_e.shape)
        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, binaries = self.vector_quantization(
            z_e)

        h = self.pre_dequantization_conv(z_q)

        x_hat = self.decoder(h)

        if verbose:
            print("z_e2", z_e.shape)
            print('z_q1', z_q.shape)
            print("z_q2", h.shape)
            print("Out ", x_hat.shape)

        if include_binaries:
            return embedding_loss, z_e, x_hat, perplexity, binaries
        else:
            return embedding_loss, z_e, x_hat, perplexity

    def encode(self, x):
        #print('encode x', x.shape)
        batch_size = x.shape[0]
        x = x.view(batch_size, self.imsize, self.imsize, self.input_channels)
        x = x.permute(0, 3, 1, 2)
        z = self.pre_quantization_conv(
            self.encoder(x))  # .detach().cpu().numpy()
        z = z.view(batch_size, -1)
        return z, z

    def decode(self, z):
        #print('decode z', z.shape)

        batch_size = z.shape[0]
        z = torch.tensor(z).float()
        z = z.view(batch_size, self.z_dim, self.z_dim, self.embedding_dim)
        z = z.permute(0, 3, 1, 2)
        x = self.pre_dequantization_conv(z)
        x = self.decoder(x)  # .detach().cpu().numpy()
        x = x.view(batch_size, -1)
        return x, x

    def log_prob(self):
        # same as vqvae loss
        raise NotImplementedError()

    def vqvae_loss(self):
        raise NotImplementedError()

    def pixelcnn_loss(self):
        raise NotImplementedError()

    def update_step(self):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((16, 3, 48, 48))
    x = torch.tensor(x).float()

    # test encoder
    # encoder = Encoder(40, 128, 3, 64)
    # encoder_out = encoder(x)
    # print('Encoder out shape:', encoder_out.shape)
    architecture = imsize48_default_architecture

    vqvae = VQVAE(architecture, 48, 64, 2, .25, 0)
    vqvae = vqvae.to(vqvae.device)
    x = x.to(vqvae.device)
    _, z, x_hat, _ = vqvae(x, verbose=True)
    print(z.shape, x_hat.shape)
    x = x[:1, :, :, :]
    z = vqvae.encode(x)
    print(z.shape)
    x_ = vqvae.decode(z)
    print(x_.shape)
