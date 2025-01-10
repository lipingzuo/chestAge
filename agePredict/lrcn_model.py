import torch.nn as nn
from torchvision import models


class ConvLstm(nn.Module):
    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional, n_class):
        super(ConvLstm, self).__init__()
        self.conv_model = Pretrained_conv(latent_dim)
        self.Lstm = Lstm(latent_dim, hidden_size, lstm_layers, bidirectional)
        self.output_layer = nn.Sequential(
            nn.Linear(2 * hidden_size if bidirectional == True else hidden_size, n_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.conv_model(conv_input)
        lstm_input = conv_output.view(batch_size, timesteps, -1)
        lstm_output = self.Lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]
        output = self.output_layer(lstm_output)
        return output


# 预训练权重偏置参数被冻结，后续的全连接层参数在训练中会更新
class Pretrained_conv(nn.Module):
    def __init__(self, latent_dim):
        super(Pretrained_conv, self).__init__()
        '''
        # resnet18
        self.conv_model = models.resnet18(weights=None)
        self.conv_model.fc = nn.Linear(self.conv_model.fc.in_features, latent_dim)
        '''
        # vgg11
        self.conv_model = models.vgg11(pretrained=False)
        self.conv_model.classifier[6] = nn.Linear(self.conv_model.classifier[6].in_features, latent_dim)


    def forward(self, x):
        return self.conv_model(x)


class Lstm(nn.Module):
    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional):
        super(Lstm, self).__init__()
        self.Lstm = nn.LSTM(latent_dim, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True,
                            bidirectional=bidirectional)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        output, self.hidden_state = self.Lstm(x, self.hidden_state)
        return output
