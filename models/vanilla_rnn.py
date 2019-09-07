import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, config, pad_idx):
        super(RNN, self).__init__()
        # add two to vocab_size for <UNK> and <PAD> tokens made by torchtext
        self.embedding = nn.Embedding(config.vocab_size + 2, config.embed_dim)
        self.rnn = nn.RNN(config.embed_dim, config.hidden_dim)
        self.fc = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, input, input_lengths):
        # input = [input length, batch size]

        embedded = self.embedding(input)
        # embedded = [input length, batch size, embed dim]

        # In PyTorch,  if no initial hidden state is passed as an argument it defaults to a tensor of all zeros
        output, hidden = self.rnn(embedded)

        # output = [input length, batch size, hidden dim]
        # : concatenation of the hidden state from every time step
        # hidden = [1, batch size, hidden dim]
        # : the 'final' hidden state

        # check whether 'hidden' is the last hidden state of 'output'
        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        # after squeeze(0), hidden becomes [batch size, hidden dim]
        return self.fc(hidden.squeeze(0))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
