import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from fdsa.models.set_matching.dnn import DNNSetMatching


class VAE(nn.Module):
    def __init__(self, params: dict, predict: bool = False) -> None:
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

    def encoder(self, x):
        x = self.encoding_layers(x)

        z_mean = self.mean_layer(x)
        z_logvar = self.var_layer(x)

        return z_mean, z_logvar

    def reparameterise(self, z_mean, z_logvar):

        z_std = torch.exp(0.5 * z_logvar)

        q_z = self.z_sampler(z_mean, z_std)

        p_z = self.prior(
            torch.full_like(z_mean, self.prior_mean),
            torch.full_like(z_std, self.prior_std),
        )

        z = q_z.rsample()

        return z, q_z, p_z

    def decoder(self, z):
        x_recon = self.decoding_layers(z)

        return x_recon

    def loss(self, x_recon, x, q_z, p_z):

        loss_recon = nn.MSELoss(reduction="none")(x_recon, x).sum(-1).mean()

        loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()

        return loss_recon + loss_KL

    def forward(self, x):

        z_mean, z_logvar = self.encoder(x)
        z, q_z, p_z = self.reparameterise(z_mean, z_logvar)
        x_recon = self.decoder(z)
        x_recon = self.final_layer(x_recon)
        if self.predict:
            prediction = self.FC_predictor(z)
            return x_recon, prediction, z, q_z, p_z
        else:
            return x_recon, z, q_z, p_z
