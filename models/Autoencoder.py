from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class AutoEncoder(nn.Module):
    def __init__(self, input_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, input_size), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoEncoder1(nn.Module):
    def __init__(self, input_size):
        super(AutoEncoder1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 5460),
            nn.ReLU(True),
            nn.Linear(5460, 1365),
            nn.ReLU(True), nn.Linear(1365, 342), nn.ReLU(True), nn.Linear(342, 86))
        self.decoder = nn.Sequential(
            nn.Linear(86, 342),
            nn.ReLU(True),
            nn.Linear(342, 1365),
            nn.ReLU(True),
            nn.Linear(1365, 5460),
            nn.ReLU(True), nn.Linear(5460, input_size), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
class AutoEncoder2(nn.Module):
    def __init__(self, input_size):
        super(AutoEncoder2, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
class ConvAutoEncoder(nn.Module):
    def __init__(self, input_size):
        super(ConvAutoEncoder, self).__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.encoder_hidden = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.decoder_hidden = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)  # 添加批次维度和通道维度
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=self.input_size)  # 调整输出大小
        x = x.squeeze(0).squeeze(0)  # 移除批次维度和通道维度
        return x
class ConvAutoEncoder_fmnist(nn.Module):
    def __init__(self, input_size, hidden_layers=5, base_channels=32):
        super(ConvAutoEncoder_fmnist, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers

        # Encoder
        encoder_layers = [
            nn.Conv1d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        ]
        in_channels = base_channels
        for _ in range(hidden_layers):
            encoder_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1))
            encoder_layers.append(nn.ReLU(True))
        self.encoder = nn.Sequential(*encoder_layers)

        # Linear layers
        self.encoder_linear = nn.Sequential(
            nn.Linear(22240, 5460),
            nn.ReLU(True),
            nn.Linear(5460, 1365),
            nn.ReLU(True),
            nn.Linear(1365, 342),
            nn.ReLU(True),
            nn.Linear(342, 86)
        )
        self.decoder_linear = nn.Sequential(
            nn.Linear(86, 342),
            nn.ReLU(True),
            nn.Linear(342, 1365),
            nn.ReLU(True),
            nn.Linear(1365, 5460),
            nn.ReLU(True),
            nn.Linear(5460, 22240)
        )

        # Decoder
        decoder_layers = []
        for _ in range(hidden_layers):
            decoder_layers.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
            decoder_layers.append(nn.ReLU(True))
        decoder_layers.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1))
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)  # 添加批次维度和通道维度
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.encoder_linear(x)  # 编码器线性层
        x = self.decoder_linear(x)  # 解码器线性层
        x = x.view(x.size(0), -1, 695)  # 恢复形状
        x = self.decoder(x)
        x = F.interpolate(x, size=self.input_size)  # 调整输出大小
        x = x.squeeze(0).squeeze(0)  # 移除批次维度和通道维度
        return x
class ConvAutoEncoder_cifar(nn.Module):
    def __init__(self, input_size, hidden_layers=11, base_channels=32):
        super(ConvAutoEncoder_cifar, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers

        # Encoder
        encoder_layers = [
            nn.Conv1d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        ]
        in_channels = base_channels
        for _ in range(hidden_layers):
            encoder_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1))
            encoder_layers.append(nn.ReLU(True))
        self.encoder = nn.Sequential(*encoder_layers)

        # Linear layers
        self.encoder_linear = nn.Sequential(
            nn.Linear(36224, 5460),
            nn.ReLU(True),
            nn.Linear(5460, 1365),
            nn.ReLU(True),
            nn.Linear(1365, 342),
            nn.ReLU(True),
            nn.Linear(342, 86)
        )
        self.decoder_linear = nn.Sequential(
            nn.Linear(86, 342),
            nn.ReLU(True),
            nn.Linear(342, 1365),
            nn.ReLU(True),
            nn.Linear(1365, 5460),
            nn.ReLU(True),
            nn.Linear(5460, 36224)
        )

        # Decoder
        decoder_layers = []
        for _ in range(hidden_layers):
            decoder_layers.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
            decoder_layers.append(nn.ReLU(True))
        decoder_layers.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1))
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)  # 添加批次维度和通道维度
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.encoder_linear(x)  # 编码器线性层
        x = self.decoder_linear(x)  # 解码器线性层
        x = x.view(x.size(0), -1, 1132)  # 恢复形状
        x = self.decoder(x)
        x = F.interpolate(x, size=self.input_size)  # 调整输出大小
        x = x.squeeze(0).squeeze(0)  # 移除批次维度和通道维度
        return x

class CAE_hidden(nn.Module):
    def __init__(self, input_size, hidden_layers=3):
        super(CAE_hidden, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers

        # Encoder
        encoder_layers = [
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        ]
        in_channels = 32
        for _ in range(hidden_layers):
            encoder_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=3, stride=2, padding=1))
            encoder_layers.append(nn.ReLU(True))
            in_channels *= 2
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for _ in range(hidden_layers):
            decoder_layers.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1))
            decoder_layers.append(nn.ReLU(True))
            in_channels //= 2
        decoder_layers.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1))
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)  # 添加批次维度和通道维度
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=self.input_size)  # 调整输出大小
        x = x.squeeze(0).squeeze(0)  # 移除批次维度和通道维度
        return x

class CAE_hidden_mp(nn.Module):
    def __init__(self, input_size, hidden_layers=3):
        super(CAE_hidden_mp, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers

        # Encoder
        encoder_layers = [
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 添加池化层
        ]
        in_channels = 32
        for _ in range(hidden_layers):
            encoder_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=3, stride=2, padding=1))
            encoder_layers.append(nn.ReLU(True))
            encoder_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))  # 添加池化层
            in_channels *= 2
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for _ in range(hidden_layers):
            decoder_layers.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1))
            decoder_layers.append(nn.ReLU(True))
            decoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))  # 添加上采样层
            in_channels //= 2
        decoder_layers.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1))
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)  # 添加批次维度和通道维度
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=self.input_size)  # 调整输出大小
        x = x.squeeze(0).squeeze(0)  # 移除批次维度和通道维度
        return x

class CAE_hidden2(nn.Module):
    def __init__(self, input_size, hidden_layers=3):
        super(CAE_hidden2, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers

        # Encoder
        encoder_layers = [
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        ]
        in_channels = 32
        for _ in range(hidden_layers):
            encoder_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1))
            encoder_layers.append(nn.ReLU(True))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for _ in range(hidden_layers):
            decoder_layers.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
            decoder_layers.append(nn.ReLU(True))
        decoder_layers.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1))
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)  # 添加批次维度和通道维度
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=self.input_size)  # 调整输出大小
        x = x.squeeze(0).squeeze(0)  # 移除批次维度和通道维度
        return x

class VAE(nn.Module):
    def __init__(self,input_size,args):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, input_size)
        self.args = args
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.FloatTensor(std.size()).normal_().to(self.args.device)
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # return F.sigmoid(self.fc4(h3))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
class VAE1(nn.Module):
    def __init__(self,input_size,args):
        super(VAE1, self).__init__()

        self.fc1 = nn.Linear(input_size, 1365)
        self.fc21 = nn.Linear(1365, 86)
        self.fc22 = nn.Linear(1365, 86)
        self.fc3 = nn.Linear(86, 1365)
        self.fc4 = nn.Linear(1365, input_size)
        self.args = args
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.FloatTensor(std.size()).normal_().to(self.args.device)
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # return F.sigmoid(self.fc4(h3))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
