from torch.nn import Sequential, Module, Conv2d, LSTM, Linear, MaxPool2d, BatchNorm2d, ReLU, Dropout
import torch
from torch.autograd import Variable


class BiLSTM(Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BiLSTM, self).__init__()

        self.rnn = LSTM(nIn, nHidden, bidirectional=True)
        self.drop = Dropout(0.5)
        self.linear = Linear(nHidden * 2, nOut)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        t, b, h = recurrent.size()
        x = recurrent.view(t * b, h)

        x = self.drop(x)

        x = self.linear(x)          # [t * b, nOut]
        x = x.view(t, b, -1)

        return x


class CRNN(Module):

    def __init__(self, im_h, n_channel, n_class, n_hidden):
        super(CRNN, self).__init__()
        assert im_h % 16 == 0, 'imgH has to be a multiple of 16'

        self.cnn1 = Sequential(
            Conv2d(n_channel, 64, 3, 1, 1),
            ReLU(True),
            MaxPool2d(2, 2)
        )

        self.cnn2 = Sequential(
            Conv2d(64, 128, 3, 1, 1),
            ReLU(True),
            MaxPool2d(2, 2)
        )

        self.cnn3 = Sequential(
            Conv2d(128, 256, 3, 1, 1),
            BatchNorm2d(256),
            ReLU(True),
            Conv2d(256, 256, 3, 1, 1),
            ReLU(True),
            MaxPool2d((2, 2), (2, 1), (0, 1))
        )

        self.cnn4 = Sequential(
            Conv2d(256, 512, 3, 1, 1),
            BatchNorm2d(512),
            ReLU(True),
            Conv2d(512, 512, 3, 1, 1),
            ReLU(True),
            MaxPool2d((2, 2), (2, 1), (0, 1))
        )

        self.cnn5 = Sequential(
            Conv2d(512, 512, 2, 1, 0),
            BatchNorm2d(512),
            ReLU(True)
        )

        self.rnn = Sequential(
            # BiLSTM(512, n_hidden, n_hidden),
            # BiLSTM(n_hidden, n_hidden, n_class)
            BiLSTM(512, n_hidden, n_class),
        )

    def forward(self, x):
        # CNN
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)

        # image -> sequence
        b, c, h, w = x.size()
        # assert h == 1, "the height of conv must be 1"
        # x = x.squeeze(2)
        x = x.view(b, c*h, w)
        x = x.permute(2, 0, 1)      # [w, b, c]

        # RNN
        x = self.rnn(x).log_softmax(2)

        return x


if __name__ == '__main__':
    net = CRNN(48, 1, 28, 256)
    image = torch.empty((10, 1, 32, 200))
    pred = net(image)

