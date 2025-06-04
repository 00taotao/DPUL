import torch
import torch.nn as nn
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


def kaiming_init(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)


def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=1, seq_length=64):  # 添加 seq_length 参数
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.seq_length = seq_length

        # 计算最终的特征长度
        def calc_out_length(L, kernel_size, stride, padding, transposed=False):
            if transposed:
                return stride * (L - 1) + kernel_size - 2 * padding
            return (L + 2 * padding - kernel_size) // stride + 1

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(nc, 32, 4, 2, 1),  # B,  32, L/2
            nn.ReLU(True),
            nn.Conv1d(32, 32, 4, 2, 1),  # B,  32, L/4
            nn.ReLU(True),
            nn.Conv1d(32, 64, 4, 2, 1),  # B,  64, L/8
            nn.ReLU(True),
            nn.Conv1d(64, 64, 4, 2, 1),  # B,  64, L/16
            nn.ReLU(True),
            nn.Conv1d(64, 256, 4, 1, 0),  # B, 256, L/16 - 3
            nn.ReLU(True),
            View((-1, 256 * (calc_out_length(calc_out_length(
                calc_out_length(calc_out_length(calc_out_length(seq_length, 4, 2, 1), 4, 2, 1), 4, 2, 1), 4, 2, 1), 4,1, 0)))),  # B, 256*(L/16 - 3)
            nn.Linear(256 * (calc_out_length(calc_out_length(calc_out_length(calc_out_length(calc_out_length(seq_length, 4, 2, 1), 4, 2, 1), 4, 2, 1), 4,2, 1), 4, 1, 0)), z_dim * 2),  # B, z_dim*2
            )

        # Decoder
        self.decoder = nn.Sequential(
        nn.Linear(z_dim, 256 * (calc_out_length(
            calc_out_length(calc_out_length(calc_out_length(calc_out_length(seq_length, 4, 2, 1), 4, 2, 1), 4, 2, 1), 4, 2,
                            1), 4, 1, 0))),  # B, 256*(L/16 - 3)
        View((-1, 256, calc_out_length(
            calc_out_length(calc_out_length(calc_out_length(calc_out_length(seq_length, 4, 2, 1), 4, 2, 1), 4, 2, 1), 4, 2,
                            1), 4, 1, 0))),  # B, 256, L/16 - 3
        nn.ReLU(True),
        nn.ConvTranspose1d(256, 64, 4),  # B,  64, L/16 + 1
        nn.ReLU(True),
        nn.ConvTranspose1d(64, 64, 4, 2, 1),  # B,  64, L/8 + 2
        nn.ReLU(True),
        nn.ConvTranspose1d(64, 32, 4, 2, 1),  # B,  32, L/4 + 4
        nn.ReLU(True),
        nn.ConvTranspose1d(32, 32, 4, 2, 1),  # B,  32, L/2 + 8
        nn.ReLU(True),
        nn.ConvTranspose1d(32, nc, 4, 2, 1),  # B, nc, L + 16
        )

        self.weight_init()


    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)


    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)  # 添加批次维度和通道维度
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x = self._decode(z)
        x = F.interpolate(x, size=self.seq_length)  # 调整输出大小
        x_recon = x.squeeze(0).squeeze(0)  # 移除批次维度和通道维度
        return x_recon, mu, logvar


    def _encode(self, x):
        return self.encoder(x)


    def _decode(self, z):
        return self.decoder(z)

