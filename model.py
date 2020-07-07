import torch
from torch import nn
from torch.nn import functional as F


import math


class Network(nn.Module):
    def __init__(self, num_actions, image_channels, vec_size, cnn_module, hidden_size=256,
                 dueling=True, double_channels=False):
        super().__init__()

        self.num_actions = num_actions
        self.dueling = dueling

        self.cnn = cnn_module(image_channels)

        self.conv_output_size = self.cnn.output_size
        self.fc_im = nn.Linear(self.conv_output_size, hidden_size)

        if not double_channels:
            vec_channel_size = 128
        else:
            vec_channel_size = 256

        self.fc_vec = nn.Linear(vec_size, vec_channel_size)

        self.fc_h_a = nn.Linear(hidden_size + vec_channel_size, hidden_size)
        
        self.fc_a = nn.Linear(hidden_size, num_actions)

        if self.dueling:
            self.fc_h_v = nn.Linear(hidden_size + vec_channel_size, hidden_size)
            self.fc_v = nn.Linear(hidden_size, 1)

    def forward(self, x, vec):
        x = self.cnn(x)
        x = x.view(-1, self.conv_output_size)
        x = self.fc_im(x)
        vec = self.fc_vec(vec)

        x = F.relu(torch.cat((x, vec), 1))

        output = self.fc_a(F.relu(self.fc_h_a(x)))
        
        if self.dueling:
            v = self.fc_v(F.relu(self.fc_h_v(x)))
            output = v + output - output.mean(1, keepdim=True)

        return output


class AtariCNN(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(input_channels, 32, 8, stride=4, padding=0),
                                         nn.ReLU(),
                                         nn.Conv2d(32, 64, 4, stride=2, padding=0),
                                         nn.ReLU(),
                                         nn.Conv2d(64, 64, 3, stride=1, padding=0),
                                         nn.ReLU())

        self.output_size = 64 * 4 * 4

    def forward(self, x):
        return self.conv_layers(x)


class ImpalaResNetCNN(nn.Module):
    class _ImpalaResidual(nn.Module):

        def __init__(self, depth):
            super().__init__()
            self.conv1 = nn.Conv2d(depth, depth, 3, padding=1)
            self.conv2 = nn.Conv2d(depth, depth, 3, padding=1)

        def forward(self, x):
            out = F.relu(x)
            out = self.conv1(out)
            out = F.relu(out)
            out = self.conv2(out)
            return out + x

    def __init__(self, input_channels):
        super().__init__()
        depth_in = input_channels
        layers = []
        for depth_out in [32, 64, 64]:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                self._ImpalaResidual(depth_out),
                self._ImpalaResidual(depth_out),
            ])
            depth_in = depth_out
        self.conv_layers = nn.Sequential(*layers, nn.ReLU())
        self.output_size = math.ceil(64 / 8) ** 2 * depth_in

    def forward(self, x):
        return self.conv_layers(x)


class FixupResNetCNN(nn.Module):
    """source: https://github.com/unixpickle/obs-tower2/blob/master/obs_tower2/model.py"""

    class _FixupResidual(nn.Module):
        def __init__(self, depth, num_residual):
            super().__init__()
            self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
            for p in self.conv1.parameters():
                p.data.mul_(1 / math.sqrt(num_residual))
            for p in self.conv2.parameters():
                p.data.zero_()
            self.bias1 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.bias2 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.bias3 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.bias4 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.scale = nn.Parameter(torch.ones([depth, 1, 1]))

        def forward(self, x):
            x = F.relu(x)
            out = x + self.bias1
            out = self.conv1(out)
            out = out + self.bias2
            out = F.relu(out)
            out = out + self.bias3
            out = self.conv2(out)
            out = out * self.scale
            out = out + self.bias4
            return out + x

    def __init__(self, input_channels, double_channels=False):
        super().__init__()
        depth_in = input_channels

        layers = []
        if not double_channels:
            channel_sizes = [32, 64, 64]
        else:
            channel_sizes = [64, 128, 128]
        for depth_out in channel_sizes:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                self._FixupResidual(depth_out, 8),
                self._FixupResidual(depth_out, 8),
            ])
            depth_in = depth_out
        layers.extend([
            self._FixupResidual(depth_in, 8),
            self._FixupResidual(depth_in, 8),
        ])
        self.conv_layers = nn.Sequential(*layers, nn.ReLU())
        self.output_size = math.ceil(64 / 8) ** 2 * depth_in

    def forward(self, x):
        return self.conv_layers(x)
