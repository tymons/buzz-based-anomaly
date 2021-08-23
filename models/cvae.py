import torch

from torch import nn

from models.ae import Decoder, Encoder
from models.vae import reparameterize


def permutate_latent(latents_batch, inplace=False):
    """ Function for element permutation along specified axis
    
    Parameters:
        latent_batch (torch.tensor): input matrix to be permutated
        inplace (bool): modify original tensor or not
    Returns
    """
    latents_batch = latents_batch.squeeze()

    data = latents_batch.detach().clone() if inplace == False else latents_batch

    for column_idx in range(latents_batch.shape[-1]):
        rand_indicies = torch.randperm(latents_batch[:, column_idx].shape[0])
        latents_batch[:, column_idx] = latents_batch[:, column_idx][rand_indicies]

    return data


def discriminator_loss(log_ratio_p, log_ratio_q):
    loss_p = nn.functional.binary_cross_entropy_with_logits(log_ratio_p, torch.ones_like(log_ratio_p), reduction='mean')
    loss_q = nn.functional.binary_cross_entropy_with_logits(log_ratio_q, torch.zeros_like(log_ratio_q),
                                                            reduction='mean')
    return loss_p + loss_q


def _kld_loss(mean, log_var):
    """ KLD loss for normal distribution"""
    return torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp())).item()


def cvae_loss(input_target, input_background, cvae_output, discriminator=None, discriminator_alpha=None, kld_weight=1):
    """
    This function will add reconstruction loss along with KLD
    :param input_tuple: (input_target, input_background)
    :param cvae_output: cvae  model output
    :param input_targetL
    :param input_background:
    """
    # MSE target
    loss = nn.functional.mse_loss(input_target, cvae_output['target'], reduction='mean')
    # MSE background
    loss += nn.functional.mse_loss(input_background, cvae_output['background'], reduction='mean')
    # KLD loss target s
    loss += kld_weight * _kld_loss(cvae_output['tg_qs_mean'], cvae_output['tg_qs_log_var'])
    # KLD loss target z
    loss += kld_weight * _kld_loss(cvae_output['tg_qz_mean'], cvae_output['tg_qz_log_var'])
    # KLD loss background z
    loss += kld_weight * _kld_loss(cvae_output['bg_qz_mean'], cvae_output['bg_qz_log_var'])

    if discriminator and discriminator_alpha:
        # total correction loss
        q = torch.cat((cvae_output["latent_qs_target"], cvae_output["latent_qz_target"]), axis=-1)
        q_score, _ = discriminator(q, torch.zeros_like(q))
        disc_loss = discriminator_alpha * torch.mean(torch.log(q_score / (1 - q_score)))
        loss += disc_loss

    return loss


class cVAE(nn.Module):
    """ Class for Contrastive autoencoder """

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, input_size):
        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list
        assert type(input_size) == int

        self.latent_size = latent_size
        self.s_encoder = Encoder(encoder_layer_sizes, input_size)
        self.z_encoder = Encoder(encoder_layer_sizes, input_size)
        self.decoder = Decoder(decoder_layer_sizes, 2 * latent_size, input_size)

        self.s_linear_means = nn.Linear(encoder_layer_sizes[-1], latent_size)
        self.s_linear_log_var = nn.Linear(encoder_layer_sizes[-1], latent_size)

        self.z_linear_means = nn.Linear(encoder_layer_sizes[-1], latent_size)
        self.z_linear_log_var = nn.Linear(encoder_layer_sizes[-1], latent_size)

    def forward(self, target, background):
        target_s = self.s_encoder(target)
        tg_s_mean, tg_s_log_var = self.s_linear_means(target_s), self.s_linear_log_var(target_s)

        target_z = self.z_encoder(target)
        tg_z_mean, tg_z_log_var = self.z_linear_means(target_z), self.z_linear_log_var(target_z)

        backgroud_z = self.z_encoder(background)
        bg_z_mean, bg_z_log_var = self.z_linear_means(backgroud_z), self.z_linear_log_var(backgroud_z)

        tg_s = reparameterize(tg_s_mean, tg_s_log_var)
        tg_z = reparameterize(tg_z_mean, tg_z_log_var)
        bg_z = reparameterize(bg_z_mean, bg_z_log_var)

        tg_output = self.decoder(torch.cat((tg_z, tg_s), axis=2))
        bg_output = self.decoder(torch.cat((bg_z, torch.zeros_like(tg_s)), axis=2))

        return {'target': tg_output,
                'tg_qs_mean': tg_s_mean,
                'tg_qs_log_var': tg_s_log_var,
                'tg_qz_mean': tg_z_mean,
                'tg_qz_log_var': tg_z_log_var,
                'background': bg_output,
                'bg_qz_mean': bg_z_mean,
                'bg_qz_log_var': bg_z_log_var,
                'latent_qs_target': tg_s,  # we need this for disentangle and ensure that s and z distributions
                'latent_qz_target': tg_z}  # for target are independent
