import torch
from torch import nn
import torch.nn.functional as F
from pytorch_revgrad import RevGrad

from .layers import TemporalConvNet
from .layers import ResidualBlock, ResidualBlock1D
from .layers import SELayer, SimAM


"""
2D Convolution Models
"""


class Conv2D(nn.Module):
    def __init__(self, configs):
        super(Conv2D, self).__init__()
        self.attention = configs.attention
        self.conv1 = nn.Sequential(
            nn.Conv2d(configs.in_channels, 16, 3),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(inplace=True)
        )
        self.se = SELayer(32, configs.reduction)
        self.simam = SimAM()
        self.adaptive_pool = nn.AdaptiveAvgPool2d(configs.feature_matrix)
        # self.fc = nn.Linear(configs.feature_matrix ** 2 * 32, 32)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)

        if self.attention == 'se':
            y = self.se(y)
        if self.attention == 'simam':
            y = self.simam(y)

        y = self.adaptive_pool(y)
        outputs = y.view(y.shape[0], -1)
        # outputs = self.fc(y)
        return outputs


class LeNet5(nn.Module):
    def __init__(self, configs):
        super(LeNet5, self).__init__()
        self.attention = configs.attention
        self.conv1 = nn.Sequential(
            nn.Conv2d(configs.in_channels, 6, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, configs.out_channels, 5),
            nn.ReLU(inplace=True),
        )
        self.se = SELayer(configs.out_channels, configs.reduction)
        self.simam = SimAM()
        self.adaptive_pool = nn.AdaptiveAvgPool2d(configs.feature_matrix)
        # self.fc = nn.Linear(configs.feature_matrix ** 2 * configs.out_channels, 32)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        if self.attention == 'se':
            y = self.se(y)
        if self.attention == 'simam':
            y = self.simam(y)

        y = self.adaptive_pool(y)
        outputs = y.view(y.shape[0], -1)
        # outputs = self.fc(y)
        return outputs


class VGG11(nn.Module):
    def __init__(self, configs):
        super(VGG11, self).__init__()
        self.encoder = self._make_layers(configs)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(configs.feature_matrix)
        # self.fc = nn.Linear(configs.feature_matrix ** 2 * self.in_channles, 32)

    def forward(self, x):
        y = self.encoder(x)
        y = self.adaptive_pool(y)
        outputs = y.view(y.shape[0], -1)
        # outputs = self.fc(y)
        return outputs

    def _make_layers(self, configs):
        layers = []
        self.in_channles = configs.in_channels
        for i in configs.VGG.cfg:
            if i == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channles, i, kernel_size=3, padding=1),
                           nn.BatchNorm2d(i),
                           nn.ReLU(inplace=True)]
                self.in_channles = i
        return nn.Sequential(*layers)


# ResNet18: number_blocks=2
class ResNet(nn.Module):
    def __init__(self, configs):
        super(ResNet, self).__init__()
        self.attention = configs.attention
        self.inchannel = configs.in_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.inchannel, configs.ResNet.cfg[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(configs.ResNet.cfg[0]),
            nn.ReLU(inplace=True)
        )
        self.inchannel = configs.ResNet.cfg[0]
        self.layer1 = self.make_layer(ResidualBlock, configs.ResNet.cfg[1], 1, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, configs.ResNet.cfg[2], 1, stride=2)
        # self.layer3 = self.make_layer(ResidualBlock, configs.ResNet.cfg[3], 1, stride=2)
        # self.layer4 = self.make_layer(ResidualBlock, configs.ResNet.cfg[4], 1, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(configs.feature_matrix)
        # self.fc = nn.Linear(configs.feature_matrix ** 2 * self.inchannel, 32)

    def make_layer(self, block, channels, num_blocks, stride, dilation=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, dilation, self.attention))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return out


# DRN: https://github.com/fyu/drn/tree/master
class DRN(nn.Module):
    def __init__(self, configs):
        super(DRN, self).__init__()
        self.attention = configs.attention
        self.inchannel = configs.in_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.inchannel, configs.ResNet.cfg[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(configs.ResNet.cfg[0]),
            nn.ReLU(inplace=True)
        )
        self.inchannel = configs.ResNet.cfg[0]
        self.layer1 = self.make_layer(ResidualBlock, configs.ResNet.cfg[1], 1, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, configs.ResNet.cfg[2], 1, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, configs.ResNet.cfg[3], 1, stride=2, dilation=2)
        self.layer4 = self.make_layer(ResidualBlock, configs.ResNet.cfg[4], 1, stride=2, dilation=4)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(configs.feature_matrix)
        # self.fc = nn.Linear(configs.feature_matrix ** 2 * self.inchannel, 32)

    def make_layer(self, block, channels, num_blocks, stride, dilation=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, dilation, self.attention))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return out


class Inception(nn.Module):
    pass


"""
1D Convolution Models
"""


class Conv1d(nn.Module):
    def __init__(self, configs):
        super(Conv1d, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=configs.in_channels, out_channels=configs.Conv1D.hidden_channels[0],
                      kernel_size=configs.Conv1D.kernel_size, stride=configs.Conv1D.stride, bias=False,
                      padding=(configs.Conv1D.kernel_size // 2)),
            nn.BatchNorm1d(configs.Conv1D.hidden_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(configs.dropout)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.Conv1D.hidden_channels[0], configs.Conv1D.hidden_channels[1],
                      kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.Conv1D.hidden_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.Conv1D.hidden_channels[1], configs.Conv1D.hidden_channels[2],
                      kernel_size=4, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(configs.Conv1D.hidden_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.feature_len)

        self.fc = nn.Linear(configs.feature_len * configs.Conv1D.hidden_channels[2], configs.feature_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        outputs = self.fc(x_flat)
        return outputs


class TCN(nn.Module):
    def __init__(self, configs):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(configs.in_channels, configs.num_channels, kernel_size=configs.kernel_size, dropout=configs.dropout)

        self.fc = nn.Linear(configs.num_channels[-1], configs.feature_len)

    def forward(self, x_in):
        """Inputs have to have dimension (N, C_in, L_in)"""
        tcn_out = self.tcn(x_in)  # input should have dimension (N, C, L)
        outputs = self.fc(tcn_out[:, :, -1])
        return outputs


class ResNet1D(nn.Module):
    def __init__(self, configs):
        super(ResNet1D, self).__init__()
        self.inchannel = configs.in_channels
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.inchannel, configs.ResNet.cfg[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(configs.ResNet.cfg[0]),
            nn.ReLU(inplace=True)
        )
        self.inchannel = configs.ResNet.cfg[0]
        self.layer1 = self.make_layer(ResidualBlock1D, configs.ResNet.cfg[1], 1, stride=1)
        self.layer2 = self.make_layer(ResidualBlock1D, configs.ResNet.cfg[2], 1, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.feature_matrix)

    def make_layer(self, block, channels, num_blocks, stride, dilation=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, dilation))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        return out


class ResNet1D3(nn.Module):
    def __init__(self, configs):
        super(ResNet1D3, self).__init__()
        self.inchannel = configs.in_channels
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.inchannel, configs.ResNet.cfg[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(configs.ResNet.cfg[0]),
            nn.ReLU(inplace=True)
        )
        self.inchannel = configs.ResNet.cfg[0]
        self.layer1 = self.make_layer(ResidualBlock1D, configs.ResNet.cfg[1], 1, stride=1)
        self.layer2 = self.make_layer(ResidualBlock1D, configs.ResNet.cfg[2], 1, stride=1)
        self.layer3 = self.make_layer(ResidualBlock1D, 64, 1, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.feature_matrix)

    def make_layer(self, block, channels, num_blocks, stride, dilation=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, dilation))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        return out


class CNN_BGM(nn.Module):
    def __init__(self, configs):
        super(CNN_BGM, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=configs.in_channels, out_channels=6, kernel_size=5, bias=False, padding=(5 // 2)),
            nn.BatchNorm1d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=5, stride=1, bias=False, padding=(5 // 2)),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(25)
        )
        self.linear_block = nn.Sequential(
            nn.Linear(25 * 16, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 32)
        )

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x_flat = x.reshape(x.shape[0], -1)
        outputs = self.linear_block(x_flat)
        return outputs


"""
MLP heads
"""


class MLP1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_pred = self.fc(x)

        if self.output_dim == 1:
            y_pred = self.sigmoid(y_pred)

        return y_pred


class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = self.hidden_fc(x)
        y_pred = self.output_fc(hidden)

        if self.output_dim == 1:
            y_pred = self.sigmoid(y_pred)

        return y_pred


class MLP3(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_fc_1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(inplace=True),
        )
        self.hidden_fc_2 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(inplace=True),
        )
        self.output_fc = nn.Linear(hidden_dim_2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden_1 = self.hidden_fc_1(x)
        hidden_2 = self.hidden_fc_2(hidden_1)
        y_pred = self.output_fc(hidden_2)

        if self.output_dim == 1:
            y_pred = self.sigmoid(y_pred)

        return y_pred


"""
MLP layers with Gradient Reversal Layer
import torch

from pytorch_revgrad import RevGrad

model = torch.nn.Sequential(
    torch.nn.Linear(10, 5),
    torch.nn.Linear(5, 2),
    RevGrad()
)
"""


class MLP2GRL(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            RevGrad()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_pred = self.fc(x)

        if self.output_dim == 1:
            y_pred = self.sigmoid(y_pred)

        return y_pred


class TSINet(nn.Module):
    def __init__(self, encoder, decoder, pre_trained=False):
        super(TSINet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        if pre_trained:
            for (name, param) in self.encoder.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.decoder(features)
        return outputs


class TSINet3(nn.Module):
    def __init__(self, encoder, classifier, discriminator, comparator):
        super(TSINet3, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.discriminator = discriminator
        self.comparator = comparator

    def forward(self, x):
        features = self.encoder(x)
        classes = self.classifier(features)
        domains = self.discriminator(features)
        cont_features = self.comparator(features)
        return classes, domains, cont_features


class CIRNet(nn.Module):
    def __init__(self, encoder, classifier):
        super(CIRNet, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        features = self.encoder(x)
        classes = self.classifier(features)
        return classes, features


class DANN(nn.Module):
    def __init__(self, encoder, classifier, discriminator):
        super(DANN, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.discriminator = discriminator

    def forward(self, x):
        features = self.encoder(x)
        domains = self.discriminator(features)
        classes = self.classifier(features)
        return classes, domains


class IEDGNet(nn.Module):
    def __init__(self, encoder, classifier, discriminator):
        super(IEDGNet, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.discriminator = discriminator

    def forward(self, x):
        features = self.encoder(x)
        domains = self.discriminator(features)
        classes = self.classifier(features)
        return classes, domains, features


class CDNet(nn.Module):
    def __init__(self, encoder1, encoder2, classifier, domain_clf):
        super(CDNet, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.classifier = classifier
        self.domain_clf = domain_clf

    def forward(self, x):
        cla_features = self.encoder1(x)
        dom_features = self.encoder1(x)
        classes = self.classifier(cla_features)
        domains = self.domain_clf(dom_features)

        return classes, domains, cla_features, dom_features

