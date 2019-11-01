import torch
import torch.nn as nn
import torch.nn.functional as F



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







# 10-way
# 2-shot

torch.manual_seed(1337)

ax = torch.rand(32, 20, 3, 28, 28, dtype=torch.float32) # support_x
ay = torch.randint(10, (32, 20)) # support_y
# ay_one_hot = F.one_hot(ay, 10)

tx = torch.rand(32, 2, 3, 28, 28, dtype=torch.float32) # target_x
ty = torch.randint(10, (32, 2)) # target_y

G = Embedding(3, 10)
lstm = nn.LSTM(10, hidden_size=30, num_layers=3, bidirectional=True)

def attention(inputs, support_x, support_y, num_classes, classification=False):
    similarities = torch.softmax(torch.cosine_similarity(support_x, inputs.unsqueeze(dim=1), dim=-1), dim=-1).unsqueeze(dim=1)
    if classification:
        support_y_pre = F.one_hot(support_y, num_classes)
    else:
        support_y_pre = support_y.unsqueeze(dim=-1)
    output = torch.bmm(similarities, support_y_pre.type(torch.float32)).squeeze(dim=1)
    
    return similarities, output

enclst = []
for i in range(ax.size(1)):
    enclst.append(G(ax[:,i,:,:, :]).unsqueeze(dim=1))

i = 0
enclst.append(G(tx[:, i, :, :, :]).unsqueeze(dim=1))

x1 = torch.cat(enclst, dim=1)
x2, (hn, cn) = lstm(x1)

similarities, output = attention(x2[:,-1,:], x2[:,:-1, :], ay, 10, True)
