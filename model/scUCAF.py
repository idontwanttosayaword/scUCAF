import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from utils.sclayers import MeanAct, DispAct


def build_network(layers, dropout_rate=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i], affine=True, ))
        net.append(nn.ELU(inplace=True))

        if i < len(layers) - 1:
            net.append(nn.Dropout(p=dropout_rate))
    return nn.Sequential(*net)


class scUCAF(torch.nn.Module):
    def __init__(
            self,
            num_cells,
            num_x1,
            num_x2,
            n_clusters,
            encoder1_layers,
            encoder2_layers,
            decoder1_layers,
            decoder2_layers,
    ):
        super().__init__()

        self.num_x1 = num_x1
        self.num_x2 = num_x2
        self.latent_dim = encoder1_layers[-1]

        self.mu = Parameter(torch.Tensor(n_clusters, self.latent_dim), requires_grad=True)

        self.encoder_x1_shared = build_network([num_x1] + encoder1_layers[:-1])
        self.encoder_x1_mu = nn.Linear(encoder1_layers[-2], self.latent_dim)
        self.encoder_x1_logvar = nn.Linear(encoder1_layers[-2], self.latent_dim)

        self.encoder_x2_shared = build_network([num_x2] + encoder2_layers[:-1])
        self.encoder_x2_mu = nn.Linear(encoder2_layers[-2], self.latent_dim)
        self.encoder_x2_logvar = nn.Linear(encoder2_layers[-2], self.latent_dim)

        self.projection_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.BatchNorm1d(self.latent_dim * 2, affine=True, ),
            nn.ELU(inplace=True),
            nn.Linear(self.latent_dim * 2, self.latent_dim)
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim, affine=True),
            nn.ELU(inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim, affine=True),
            nn.ELU(inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.decoder_x1_zinb = build_network(decoder1_layers)
        self.decoder_x2_zinb = build_network(decoder2_layers)

        decoder1_out_dim = decoder1_layers[-1]
        decoder2_out_dim = decoder2_layers[-1]

        self.dec_mean1 = nn.Sequential(nn.Linear(decoder1_out_dim, self.num_x1), MeanAct())
        self.dec_disp1 = nn.Sequential(nn.Linear(decoder1_out_dim, self.num_x1), DispAct())
        self.dec_pi1 = nn.Sequential(nn.Linear(decoder1_out_dim, self.num_x1), nn.Sigmoid())

        self.dec_mean2 = nn.Sequential(nn.Linear(decoder2_out_dim, self.num_x2), MeanAct())
        self.dec_disp2 = nn.Sequential(nn.Linear(decoder2_out_dim, self.num_x2), DispAct())
        self.dec_pi2 = nn.Sequential(nn.Linear(decoder2_out_dim, self.num_x2), nn.Sigmoid())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_x1(self, x):
        h = self.encoder_x1_shared(x)
        return self.encoder_x1_mu(h), self.encoder_x1_logvar(h)

    def encode_x2(self, x):
        h = self.encoder_x2_shared(x)
        return self.encoder_x2_mu(h), self.encoder_x2_logvar(h)

    def forward_zinb(self, x1, x2):
        x1_tensor = x1 + torch.randn_like(x1) * 2.5
        x2_tensor = x2 + torch.randn_like(x2) * 1.5

        mu1, logvar1 = self.encode_x1(x1_tensor)
        mu2, logvar2 = self.encode_x2(x2_tensor)

        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)

        z_fused, w1, w2 = self.variance_fusion(z1, z2, logvar1, logvar2)

        h1 = self.decoder_x1_zinb(z_fused)
        mean1 = self.dec_mean1(h1)
        disp1 = self.dec_disp1(h1)

        h2 = self.decoder_x2_zinb(z_fused)
        mean2 = self.dec_mean2(h2)
        disp2 = self.dec_disp2(h2)

        mu1_clean, logvar1_clean = self.encode_x1(x1)
        mu2_clean, logvar2_clean = self.encode_x2(x2)
        z1_clean = self.reparameterize(mu1_clean, logvar1_clean)
        z2_clean = self.reparameterize(mu2_clean, logvar2_clean)

        z_clean_fused, w1, w2 = self.variance_fusion(z1_clean, z2_clean, logvar1_clean, logvar2_clean)

        return z_clean_fused, mean1, disp1, mean2, disp2, mu1, logvar1, mu2, logvar2

    def forward(self, x1, x2):
        x1_tensor = x1 + torch.randn_like(x1) * 2.5
        x2_tensor = x2 + torch.randn_like(x2) * 1.5

        mu1, logvar1 = self.encode_x1(x1_tensor)
        mu2, logvar2 = self.encode_x2(x2_tensor)

        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)

        z_fused, w1, w2 = self.variance_fusion(z1, z2, logvar1, logvar2)

        p1 = self.projection_head(z1)
        p2 = self.projection_head(z2)

        h1 = self.decoder_x1_zinb(z_fused)
        mean1 = self.dec_mean1(h1)
        disp1 = self.dec_disp1(h1)

        h2 = self.decoder_x2_zinb(z_fused)
        mean2 = self.dec_mean2(h2)
        disp2 = self.dec_disp2(h2)

        mu1_clean, logvar1_clean = self.encode_x1(x1)
        mu2_clean, logvar2_clean = self.encode_x2(x2)
        z1_clean = self.reparameterize(mu1_clean, logvar1_clean)
        z2_clean = self.reparameterize(mu2_clean, logvar2_clean)

        z_clean_fused, w1, w2 = self.variance_fusion(z1_clean, z2_clean, logvar1_clean, logvar2_clean)

        return z_clean_fused, mean1, disp1, mean2, disp2, mu1, logvar1, mu2, logvar2, p1, p2, w1, w2

    def target_distribution(self, q):
        p = q ** 2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def clustering_loss(self, z):
        distances = torch.sum(torch.square(z.unsqueeze(1) - self.mu), dim=2)

        alpha = 1
        q = 1.0 / (1.0 + distances / alpha)
        q = torch.pow(q, (alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()

        weight = q ** 2 / torch.sum(q, dim=0)
        p = (weight.t() / torch.sum(weight, dim=1)).t()

        cluster_loss = torch.mean(torch.sum(p * torch.log(p / q), dim=1))

        return distances, cluster_loss

    def info_nce_loss(self, z1, z2, neg_weights=None):
        batch_size = z1.shape[0]

        if neg_weights is None:
            neg_weights = torch.ones(batch_size, batch_size, device=z1.device)

        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)

        logits = torch.mm(z1_norm, z2_norm.t())

        pos_mask = torch.eye(batch_size, device=z1.device)
        neg_mask = 1 - pos_mask

        weighted_logits = logits * pos_mask + logits * neg_mask * neg_weights

        labels = torch.arange(batch_size, device=z1.device)

        loss_i = F.cross_entropy(weighted_logits, labels)
        loss_j = F.cross_entropy(weighted_logits.t(), labels)

        contrastive_loss = (loss_i + loss_j) / 2

        return contrastive_loss

    def calculate_weights(self, logvar1, logvar2):
        var1 = torch.exp(logvar1) + 1e-8
        var2 = torch.exp(logvar2) + 1e-8

        total_var1 = torch.sum(var1, dim=1)
        total_var2 = torch.sum(var2, dim=1)

        w1 = total_var2 / (total_var1 + total_var2)
        w2 = total_var1 / (total_var1 + total_var2)

        return w1.unsqueeze(1), w2.unsqueeze(1)

    def variance_gate(self, logvar):
        variance = torch.exp(logvar)

        neg_variance = -variance

        gate = F.softmax(neg_variance, dim=1)

        gate = gate * self.latent_dim

        return gate

    def variance_fusion(self, z1, z2, logvar1, logvar2):
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)

        w1, w2 = self.calculate_weights(logvar1, logvar2)

        z_fused = w1 * z1 + w2 * z2

        var_fused = (w1 ** 2) * var1 + (w2 ** 2) * var2

        gate = self.variance_gate(var_fused)

        z_final = z_fused * gate

        uncertainty1 = torch.mean(var1, dim=1)
        uncertainty2 = torch.mean(var2, dim=1)

        return z_final, uncertainty1, uncertainty2

    def cosine_similarity(self, x1, x2):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(x1, x2)

    def encodeBatch(self, X1, X2, batch_size=256, plot=False, epoch=1, run_dir="", dropout_rate=0):
        z_list = []
        uncertainty1_list = []
        uncertainty2_list = []
        num_samples = X1.shape[0]
        num_batch = int(math.ceil(1.0 * num_samples / batch_size))

        self.eval()
        with torch.no_grad():
            for batch_idx in range(num_batch):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)

                x1_batch = X1[start_idx:end_idx]
                x2_batch = X2[start_idx:end_idx]

                z_clean_fused, mean1, disp1, mean2, disp2, mu1, logvar1, mu2, logvar2, p1, p2, uncertainty1, uncertainty2 = self.forward(
                    x1_batch, x2_batch)

                z_list.append(z_clean_fused)
                uncertainty1_list.append(uncertainty1)
                uncertainty2_list.append(uncertainty2)

            z_combined = torch.cat(z_list, dim=0)

            uncertainty1_combined = torch.cat(uncertainty1_list, dim=0)
            uncertainty2_combined = torch.cat(uncertainty2_list, dim=0)

            z_numpy = z_combined.detach().cpu().numpy()
            uncertainty1_numpy = uncertainty1_combined.detach().cpu().numpy()
            uncertainty2_numpy = uncertainty2_combined.detach().cpu().numpy()

            return z_numpy
