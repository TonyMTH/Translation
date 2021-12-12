import torch
from torch import nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.net = nn.RNN(hidden_size, hidden_size)

    def forward(self, x):
        hn = torch.zeros(1, 1, self.hidden_size, device=x.device)
        x = self.embedding(x).view(1, 1, -1)
        x, hn = self.net(x, hn)
        return x, hn


class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.net = nn.GRU(hidden_size, hidden_size)

    def forward(self, x):
        hn = torch.zeros(1, 1, self.hidden_size, device=x.device)
        x = self.embedding(x).view(1, 1, -1)
        x, hn = self.net(x, hn)
        return x, hn


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.net = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)

    def forward(self, x):
        hn = torch.zeros(1, 1, self.hidden_size, device=x.device)
        cn = torch.zeros(1, 1, self.hidden_size, device=x.device)
        x = self.embedding(x).view(1, 1, -1)
        x, (hn, cn) = self.net(x, (hn.detach(), cn.detach()))
        return x, hn


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.net = nn.RNN(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hn):
        x = self.embedding(x).view(1, 1, -1)
        x = F.relu(x)
        x, hn = self.net(x, hn)
        x = self.softmax(self.out(x[0]))
        return x, hn


class DecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.net = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hn):
        x = self.embedding(x).view(1, 1, -1)
        x = F.relu(x)
        x, hn = self.net(x, hn)
        x = self.softmax(self.out(x[0]))
        return x, hn


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.net = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hn):
        cn = torch.zeros(1, 1, self.hidden_size, device=x.device)
        x = self.embedding(x).view(1, 1, -1)
        x = F.relu(x)
        x, (hn, cn) = self.net(x, (hn.detach(), cn.detach()))
        x = self.softmax(self.out(x[0]))
        return x, hn


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.net = nn.RNN(self.hidden_size, self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hn, encoder_outputs):
        x = self.embedding(x).view(1, 1, -1)
        x = self.dropout(x)

        attn_weights = F.softmax(
            self.attn(torch.cat((x[0], hn[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        x = torch.cat((x[0], attn_applied[0]), 1)
        x = self.attn_combine(x).unsqueeze(0)

        x = F.relu(x)
        x, hn = self.net(x, hn)

        x = F.log_softmax(self.out(x[0]), dim=1)
        return x, hn, attn_weights


class AttnDecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.net = nn.GRU(self.hidden_size, self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hn, encoder_outputs):
        x = self.embedding(x).view(1, 1, -1)
        x = self.dropout(x)

        attn_weights = F.softmax(
            self.attn(torch.cat((x[0], hn[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        x = torch.cat((x[0], attn_applied[0]), 1)
        x = self.attn_combine(x).unsqueeze(0)

        x = F.relu(x)
        x, hn = self.net(x, hn)

        x = F.log_softmax(self.out(x[0]), dim=1)
        return x, hn, attn_weights


class AttnDecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.net = nn.LSTM(self.hidden_size, self.hidden_size, 1, batch_first=True)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hn, encoder_outputs):
        cn = torch.zeros(1, 1, self.hidden_size, device=x.device)
        x = self.embedding(x).view(1, 1, -1)
        x = self.dropout(x)

        attn_weights = F.softmax(
            self.attn(torch.cat((x[0], hn[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        x = torch.cat((x[0], attn_applied[0]), 1)
        x = self.attn_combine(x).unsqueeze(0)

        x = F.relu(x)
        x, (hn, cn) = self.net(x, (hn.detach(), cn.detach()))

        x = F.log_softmax(self.out(x[0]), dim=1)
        return x, hn, attn_weights
