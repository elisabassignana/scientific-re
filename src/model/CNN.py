import torch
from torch import nn
import torch.nn.functional as F


class CNN(torch.nn.Module):

    def __init__(self, params):
        super().__init__()

        self.device = params.device
        self.dropout = nn.Dropout(params.dropout)

        self.embedding_size = params.bert_emb_size + 2 * params.position_emb_size

        # Embedding layers definition
        self.emb_pos = nn.Embedding(params.num_positions, params.position_emb_size, padding_idx=0)

        # Convolution layers definition
        self.conv_1 = nn.Conv1d(in_channels=self.embedding_size, out_channels=params.out_size, kernel_size=params.kernels[0])
        self.conv_2 = nn.Conv1d(in_channels=self.embedding_size, out_channels=params.out_size, kernel_size=params.kernels[1])
        self.conv_3 = nn.Conv1d(in_channels=self.embedding_size, out_channels=params.out_size, kernel_size=params.kernels[2])

        # Fully connected layer definition
        self.fc = nn.Linear(3 * params.out_size, len(params.relations))

    def forward(self, w, p1, p2):

        # Prepare the input from the embeddings layers
        pos1_emb = self.emb_pos(p1)
        pos2_emb = self.emb_pos(p2)

        x = torch.cat((w, pos1_emb, pos2_emb), 2)
        x = x.permute(0,2,1)

        # Convolution layer 1 is applied
        x1 = torch.relu(self.conv_1(x))
        x1 = F.max_pool1d(x1, kernel_size=x1.size(2)).squeeze(2)

        # Convolution layer 2 is applied
        x2 = torch.relu(self.conv_2(x))
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)

        # Convolution layer 3 is applied
        x3 = torch.relu(self.conv_3(x))
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)

        # Linear layer
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.fc(self.dropout(out))

        return out