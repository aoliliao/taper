import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor

class MLP(nn.Module):
    def __init__(self, in_features=1024, out_features=4, num_layer=3, layer_list=[1000, 500, 100], dropout=0.5):
        super(MLP, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(in_features, layer_list[0]),
            # nn.BatchNorm1d(layer_list[0]),
            nn.ReLU()
        )

        self.hidden_layer = nn.Sequential()

        for index in range(num_layer - 1):
            self.hidden_layer.extend([nn.Linear(layer_list[index], layer_list[index + 1]),
                                      # nn.BatchNorm1d(layer_list[index + 1]),
                                      nn.ReLU()])
        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Sequential(
            nn.Linear(layer_list[-1], out_features,),
            nn.Softmax(dim=1),
        )


    def forward(self, x):
        input = self.input_layer(x)
        hidden = self.hidden_layer(input)
        output = self.output_layer(hidden)
        return output


class ConvNet(nn.Module):
    def __init__(self, ws=[16, 32]):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, ws[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(ws[0], ws[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ws[1], 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ConvNet_100(nn.Module):
    def __init__(self, ws=[16, 32]):
        super(ConvNet_100, self).__init__()
        self.conv1 = nn.Conv2d(3, ws[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(ws[0], ws[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ws[1], 100)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class fastText(nn.Module):
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)

        # Hidden Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        # Output Layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        out = F.log_softmax(z, dim=1)

        return out
