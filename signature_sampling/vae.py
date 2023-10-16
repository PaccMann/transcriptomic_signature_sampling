import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from typing import Tuple
import torch.distributions.distribution as distribution
from fdsa.models.set_matching.dnn import DNNSetMatching


class VAE(nn.Module):
    """Pytorch module for variational autoencoder (VAE)."""

    def __init__(self, params: dict, predict: bool = False) -> None:
        """Constructor.

        Args:
            params (dict): Dictionary of parameters describing the encoder and decoder
                of the vae, including training parameters.
            predict (bool, optional): Adds a separate prediction layer after the encoder.
                If True, specify parameters in the params dict. Defaults to False.
        """
        super().__init__()
        self.params = params
        self.predict = predict

        self.latent_dim = params["latent_size"]

        self.input_size = params["input_size"]

        self.h_size = params["fc_units"]

        self.mean_layer = nn.Linear(self.h_size[-1], self.latent_dim)
        self.var_layer = nn.Linear(self.h_size[-1], self.latent_dim)

        self.encoding_layers = DNNSetMatching(self.params)
        if self.predict:
            self.FC_predictor = DNNSetMatching(self.params)

        self.prior_params = self.params.get("prior_params")
        self.prior_mean = torch.tensor(self.prior_params.get("loc", 0.0))
        self.prior_std = torch.tensor(self.prior_params.get("scale", 1.0))

        self.prior = Normal
        self.z_sampler = Normal

        self.h_size.reverse()
        self.dec_params = params.copy()
        self.dec_params.update({"input_size": self.latent_dim, "fc_units": self.h_size})
        self.decoding_layers = DNNSetMatching(self.dec_params)

        self.final_layer = nn.Linear(self.h_size[-1], self.input_size)

    def encoder(self, x: torch.Tensor) -> Tuple:
        """Encoder function of the VAE.

        Args:
            x (torch.Tensor): Input to the encoder of shape (batch size, feature size).

        Returns:
            Tuple: Tuple of the mean and log variance of the latent features of shape
                (latent size,).
        """
        x = self.encoding_layers(x)

        z_mean = self.mean_layer(x)
        z_logvar = self.var_layer(x)

        return z_mean, z_logvar

    def reparameterise(self, z_mean: torch.Tensor, z_logvar: torch.Tensor) -> Tuple:
        """Reparametrisation trick.

        Args:
            z_mean (torch.Tensor): Vector of means of the latent features.
            z_logvar (torch.Tensor): Vector of log variances of the latent features.

        Returns:
            Tuple: Tuple of the latent embedding, posterior probability and prior
                probability distributions.
        """
        z_std = torch.exp(0.5 * z_logvar)

        q_z = self.z_sampler(z_mean, z_std)

        p_z = self.prior(
            torch.full_like(z_mean, self.prior_mean),
            torch.full_like(z_std, self.prior_std),
        )

        z = q_z.rsample()

        return z, q_z, p_z

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        """Decoder function of the VAE.

        Args:
            z (torch.Tensor): Latent embedding from the encoder of the vae.

        Returns:
            torch.Tensor: Reconstructed input from the latent embedding.
        """
        x_recon = self.decoding_layers(z)

        return x_recon

    def loss(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        q_z: distribution,
        p_z: distribution,
    ) -> torch.scalar_tensor:
        """ELBO loss for VAE training.

        Args:
            x_recon (torch.Tensor): Reconstructed input from the latent embedding.
            x (torch.Tensor): Input to the encoder of shape (batch size, feature size).
            q_z (distribution): Estimated posterior distribution from the latent space
                parameters.
            p_z (distribution): Prior distribution to approximate the posterior.

        Returns:
            torch.scalar_tensor: Average loss of the current batch of inputs.
        """
        loss_recon = nn.MSELoss(reduction="none")(x_recon, x).sum(-1).mean()

        loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()

        return loss_recon + loss_KL

    def forward(self, x: torch.Tensor) -> Tuple:
        """Forward function of the VAE class.

        Args:
            x (torch.Tensor): Input to the VAE of shape (batch size, feature size).

        Returns:
            Tuple: Tuple of reconstructed inputs, latent embeddings, posterior and
                prior distributions. Includes predictions on latent embedding if predict=True.
        """
        z_mean, z_logvar = self.encoder(x)
        z, q_z, p_z = self.reparameterise(z_mean, z_logvar)
        x_recon = self.decoder(z)
        x_recon = self.final_layer(x_recon)
        if self.predict:
            prediction = self.FC_predictor(z)
            return x_recon, prediction, z, q_z, p_z
        else:
            return x_recon, z, q_z, p_z
