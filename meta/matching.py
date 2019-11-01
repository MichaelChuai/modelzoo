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
        similarities = torch.softmax(torch.cosine_similarity(support_x, inputs.unsqueeze(dim=1), dim=-1), dim=-1)
        if self.classification:
            support_y_pre = F.one_hot(support_y, self.num_classes)
        else:
            support_y_pre = support_y.unsqueeze(dim=-1)
        output = torch.bmm(similarities.unsqueeze(dim-1), support_y_pre.type(torch.float32)).squeeze(dim=1)
        return similarities, output

    def forward(self, inputs, support_x, support_y):
        
        
