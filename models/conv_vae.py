from torch import nn
from vae import reparameterize
from conv_ae import ConvolutionalDecoder, ConvolutionalEncoder, conv_mlp_layer_shape


class ConvolutionalVAE(nn.Module):
    """ Class for convolutional variational autoencoder """

    def __init__(self, encoder_conv_sizes, encoder_mlp_sizes,
                 decoder_conv_sizes, decoder_mlp_sizes, latent_size, input_shape):
        assert type(encoder_conv_sizes) == list
        assert type(encoder_mlp_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_conv_sizes) == list
        assert type(decoder_mlp_sizes) == list
        assert type(input_shape) == tuple

        super().__init__()

        conv_to_mlp_shape = conv_mlp_layer_shape(input_shape, encoder_conv_sizes, kernel=3, stride=1, padding=1,
                                                 max_pool=(2, 2))

        self.encoder = ConvolutionalEncoder(encoder_conv_sizes, encoder_mlp_sizes, connector_shape=conv_to_mlp_shape)
        self.decoder = ConvolutionalDecoder(decoder_conv_sizes, decoder_mlp_sizes, latent_size,
                                            connector_shape=conv_to_mlp_shape)
        self.linear_means = nn.Linear(encoder_mlp_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(encoder_mlp_sizes[-1], latent_size)
        self.latent_size = latent_size

    def forward(self, x):
        y = self.encoder(x)
        means = self.linear_means(y)
        log_var = self.linear_log_var(y)
        latent = reparameterize(means, log_var)
        recon_x = self.decoder(latent)

        return {'target': recon_x, 'mean': means, 'logvar': log_var}

    def inference(self, x):
        z = self.encoder(x)
        z_mean, z_var = self.linear_means(z), self.linear_log_var(z)
        return reparameterize(z_mean, z_var)
