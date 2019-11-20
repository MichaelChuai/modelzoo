import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchingNetwork(nn.Module):
    def __init__(self, embedding_network, context_embedding_network, num_classes):
        super(MatchingNetwork, self).__init__()
        self.embedding_network = embedding_network
        self.context_embedding_network = context_embedding_network
        self.num_classes = num_classes
        if self.num_classes == -1:
            self.classification = False
        else:
            self.classification = True
        self.similarities = None

    @classmethod
    def build_context_embedding_network(cls, in_dim, out_dim, num_layers, bidirectional=True):
        network = nn.LSTM(in_dim, hidden_size=out_dim, num_layers=num_layers, bidirectional=bidirectional)
        return network

    def attention(self, inputs, support_x, support_y):
        similarities = torch.softmax(torch.cosine_similarity(support_x, inputs.unsqueeze(dim=1), dim=-1), dim=-1).unsqueeze(dim=1)
        if self.classification:
            support_y_pre = F.one_hot(support_y, self.num_classes)
        else:
            support_y_pre = support_y.unsqueeze(dim=-1)
        output = torch.bmm(similarities, support_y_pre.type(torch.float32))
        return similarities, output

    def forward(self, inputs, support_x, support_y):
        support_length = support_x.size(1)
        enclst = []
        for i in range(support_length):
            enclst.append(self.embedding_network(support_x[:,i,...]).unsqueeze(dim=1))
        encoded_support = torch.cat(enclst, dim=1)
        inputs_length = inputs.size(1)
        
        similarities_lst = []
        output_lst = []
        for i in range(inputs_length):
            encoded_target = self.embedding_network(inputs[:, i, ...]).unsqueeze(dim=1)
            context_set = torch.cat([encoded_support, encoded_target], dim=1)
            context_embedding, _ = self.context_embedding_network(context_set)
            similarities_s, output_s = self.attention(context_embedding[:, -1, ...], context_embedding[:, :-1, ...], support_y)
            similarities_lst.append(similarities_s)
            output_lst.append(output_s)

        similarities = torch.cat(similarities_lst, dim=1)
        output = torch.cat(output_lst, dim=1)
        self.similarities = similarities
        return output

            
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
# 1-shot



# G = Embedding(1, 10)
# context_embedding_network = MatchingNetwork.build_context_embedding_network(10, 64, 1)

# network = MatchingNetwork(G, context_embedding_network, 10)

# output = network(tx, ax, ay)

# loss = nn.CrossEntropyLoss()(output.transpose(-2, -1), ty)