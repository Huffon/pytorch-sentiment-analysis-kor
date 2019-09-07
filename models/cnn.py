import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(self.config.vocab_size + 2, self.config.embed_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=self.config.n_filters,
                      kernel_size=(fs, self.config.embed_dim)) for fs in self.config.filter_sizes
        ])

        # FC layer takes (filer_sizes * n_filters) size of concatenated vector as an input
        self.fc = nn.Linear(len(self.config.filter_sizes) * config.n_filters, self.config.output_dim)

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, input, input_length):
        # input = [input length, batch size]

        # In PyTorch, RNNs want the input with the batch dimension second, whereas CNNs want the batch dimension first.
        # thus, the first thing we do to our input is permute it to make it the correct shape.
        input = input.permute(1, 0)

        # input = [batch size, input length]

        embedded = self.embedding(input)
        # embedded = [batch size, input length, embed dim]

        # add additional dimension using unsqueeze(1) which takes in_channel as an input
        # image processing needs 3 in-channels (RGB), but text processing needs only one in-channel
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, input length, embed dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # by squeezing additional dimension, conved_n = [batch size, n_filters, input length - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # by squeezing additional dimension, pooled_n = [batch_size, n_filters]

        # concatenate max pooled vectors into a single vector and pass it through a final FC layer
        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(cat)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
