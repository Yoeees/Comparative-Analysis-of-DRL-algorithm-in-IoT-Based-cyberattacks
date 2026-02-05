# ConvAE.py
import torch
import torch.nn as nn


class ConvAE(nn.Module):
    def __init__(self, feat_dim, seq_len, latent_dim=128):
        super(ConvAE, self).__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(feat_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(seq_len // 2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(seq_len // 4),
        )
        self.encoder_out_dim = 64 * (seq_len // 4)
        self.bottleneck = nn.Linear(self.encoder_out_dim, latent_dim)

        # Decoder
        self.decoder_linear = nn.Linear(latent_dim, self.encoder_out_dim)
        self.decoder = nn.Sequential(
            nn.Upsample(size=seq_len // 2, mode="nearest"),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(size=seq_len, mode="nearest"),
            nn.Conv1d(32, feat_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, feat_dim, seq_len)
        encoded = self.encoder(x)
        encoded_flat = encoded.view(encoded.size(0), -1)
        latent = self.bottleneck(encoded_flat)
        decoded_flat = self.decoder_linear(latent)
        decoded = decoded_flat.view(encoded.size(0), 64, self.seq_len // 4)
        decoded = self.decoder(decoded)
        return decoded.transpose(1, 2)  # (batch, seq_len, feat_dim)

    def encode(self, x):
        x = x.transpose(1, 2)
        encoded = self.encoder(x)
        encoded_flat = encoded.view(encoded.size(0), -1)
        return self.bottleneck(encoded_flat)  # (batch, latent_dim)
