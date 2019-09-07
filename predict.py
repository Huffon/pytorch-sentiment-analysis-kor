import pickle
import argparse
import torch

from soynlp.tokenizer import LTokenizer
from models.vanilla_rnn import RNN
from models.bidirectional_lstm import BidirectionalLSTM
from models.cnn import CNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_cnn(config):
    # load tokenizer and torchtext Field
    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores)

    pickle_vocab = open('pickles/text.pickle', 'rb')
    text = pickle.load(pickle_vocab)

    model = CNN(config)
    model.load_state_dict(torch.load(config.save_model))
    model.eval()

    tokenized = tokenizer.tokenize(config.input)

    min_len = config.filter_sizes[-1]

    # if user's input sentence is shorter than the largest filter size, add pad tokens to input sentence
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))

    indexed = [text.vocab.stoi[token] for token in tokenized]
    length = [len(indexed)]

    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)

    prediction = torch.sigmoid(model(tensor, length_tensor))
    label = torch.round(prediction)

    if label == 1:
        label = 'Positive'
    else:
        label = 'Negative'

    sentiment_percent = prediction.item()
    print(f'{config.input} is {sentiment_percent * 100:.2f} % : {label}')


def predict_sequential(config):
    # load tokenizer and torchtext Field
    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores)

    pickle_vocab = open('pickles/text.pickle', 'rb')
    text = pickle.load(pickle_vocab)
    pad_idx = text.vocab.stoi[text.pad_token]

    model_type = {
        'vanilla_rnn': RNN(config, pad_idx),
        'bidirectional_lstm': BidirectionalLSTM(config, pad_idx),
    }

    # select model and load trained model
    model = model_type[config.model]
    model.load_state_dict(torch.load(config.save_model))
    model.eval()

    # convert input into tensor and forward it through selected model
    tokenized = tokenizer.tokenize(config.input)
    indexed = [text.vocab.stoi[token] for token in tokenized]
    length = [len(indexed)]

    tensor = torch.LongTensor(indexed).to(device)    # [input length]
    tensor = tensor.unsqueeze(1)                     # [input length, 1] for adding batch dimension
    length_tensor = torch.LongTensor(length)

    prediction = torch.sigmoid(model(tensor, length_tensor))
    label = torch.round(prediction)

    if label == 1:
        label = 'Positive'
    else:
        label = 'Negative'

    sentiment_percent = prediction.item()
    print(f'{config.input} is {sentiment_percent*100:.2f} % : {label}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentiment Analysis Prediction')

    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--vocab_size', type=int, default=25000)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--bidirectional', type=bool, default=True)

    parser.add_argument('--n_filters', type=int, default=100)
    parser.add_argument('--filter_sizes', type=list, default=[3, 4, 5])

    parser.add_argument('--model', type=str, default='vanilla_rnn', choices=['vanilla_rnn', 'bidirectional_lstm', 'cnn'])
    parser.add_argument('--input', type=str, default='이 영화 진짜 최고에요')
    parser.add_argument('--save_model', type=str, default='model.pt')

    config = parser.parse_args()

    if config.model == 'cnn':
        predict_cnn(config)
    else:
        predict_sequential(config)
