import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BidirectionalLSTM(nn.Module):
    def __init__(self, config, pad_idx):
        super(BidirectionalLSTM, self).__init__()
        # pass padding_idx to model not to train padding tokens
        self.embedding = nn.Embedding(config.vocab_size + 2, config.embed_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(config.embed_dim,
                            config.hidden_dim,
                            num_layers=config.n_layer,
                            bidirectional=config.bidirectional,
                            dropout=config.dropout)

        # Bidirectional models use concatenated hidden state (forward + backward),
        # therefore input dimension of FC layer has to be multiplied by 2.
        self.fc = nn.Linear(config.hidden_dim * 2, config.output_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input, input_lengths):
        # input = [input length, batch size]

        embedded = self.embedding(input)
        # embedded = [input length, batch size, embed dim]

        # pack embedded sequence using input lengths to let the model only process non-padded elements
        packed_embedded = pack_padded_sequence(embedded, input_lengths)
        # by using packed sequence, we can use last non-padded element unless we might use pad token as a last element
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # (optional) unpack sequence
        output, output_lengths = pad_packed_sequence(packed_output)
        # output = [input length, batch size, hidden dim * num directions] (padding tokens of output are zero tensors)

        # hidden, cell = [num layers * num directions, batch size, hidden dim]
        # [forward_layer_0, backward_layer_0, forward_layer_1, backward_layer_1, ..., forward_layer_n, backward_layer_n]

        # concat the final forward (hidden[-2, :, :]) and backward (hidden[-1, :, :]) hidden states
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden = [batch size, hidden dim * num directions]

        return self.fc(hidden)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
