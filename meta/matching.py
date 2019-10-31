import torch
import torch.nn as nn




class Embedding(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.1):
        super(Embedding, self).__init__()
        self.layer1 = self.convLayer(in_channels, out_channels, dropout_prob)
        self.layer2 = self.convLayer(out_channels, out_channels, dropout_prob)
        self.layer3 = self.convLayer(out_channels, out_channels, dropout_prob)
        self.layer4 = self.convLayer(out_channels, out_channels, dropout_prob)

    @classmethod
    def convLayer(cls, in_channels, out_channels, dropout_prob):
        seq = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(dropout_prob)
        )
        return seq

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        return x






lstm = nn.LSTM(10, hidden_size=30, num_layers=3, bidirectional=True)
G = Embedding(5, 10)

# 10-way
# 2-shot

ax = torch.rand(32, 20, 5, 28, 28, dtype=torch.float32) # support_x
ay = torch.randint(10, (32, 20))

# enclst = []
# for i in range(a.size(1)):
#     enclst.append(G(a[:,i,:,:]).unsqueeze(dim=1))
# x1 = torch.cat(enclst, dim=1)
# x2, (hn, cn) = lstm(x1)


# support_set = torch.rand(32, 10, 64, dtype=torch.float32)

# input_image = torch.rand(32, 64)
# sim = torch.cosine_similarity(support_set, input_image.unsqueeze(dim=1), dim=-1)
