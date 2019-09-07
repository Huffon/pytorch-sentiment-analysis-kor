import os
import pickle
import argparse
from pathlib import Path

import torch
import pandas as pd

from utils import convert_to_dataset
from torchtext import data as ttd
from soynlp.tokenizer import LTokenizer
from sklearn.model_selection import train_test_split
from soynlp.word import WordExtractor


def build_tokenizer():
    """
    Train soynlp tokenizer which will be used to tokenize Korean input sentence
    Returns:

    """
    print(f'Now building soynlp tokenizer . . .')

    data_dir = Path().cwd() / 'data'
    train_txt = os.path.join(data_dir, 'train.txt')

    with open(train_txt, encoding='utf-8') as f:
        lines = f.readlines()

    word_extractor = WordExtractor(min_frequency=5)
    word_extractor.train(lines)

    word_scores = word_extractor.extract()
    cohesion_scores = {word: score.cohesion_forward for word, score in word_scores.items()}

    with open('pickles/tokenizer.pickle', 'wb') as pickle_out:
        pickle.dump(cohesion_scores, pickle_out)


def build_vocab(config):
    """
    Build vocab used to convert Korean input sentence into word indices using soynlp tokenizer
    Args:
        config: configuration object containing various options

    Returns:

    """

    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores)

    # To use packed padded sequences, tell the model how long the actual sequences are by 'include_lengths=True'
    text = ttd.Field(tokenize=tokenizer.tokenize, include_lengths=True)
    label = ttd.LabelField(dtype=torch.float)

    data_dir = Path().cwd() / 'data'
    train_txt = os.path.join(data_dir, 'train.txt')

    train_data = pd.read_csv(train_txt, sep='\t')
    train_data, valid_data = train_test_split(train_data, test_size=0.3, random_state=32)
    train_data = convert_to_dataset(train_data, text, label)

    print(f'Building vocabulary using torchtext . . .')
    text.build_vocab(train_data, max_size=config.vocab_size)
    label.build_vocab(train_data)

    print(f'Unique tokens in TEXT vocabulary: {len(text.vocab)}')
    print(f'Unique tokens in LABEL vocabulary: {len(label.vocab)}')

    print(f'Most commonly used words are as follows:')
    print(text.vocab.freqs.most_common(20))

    file_text = open('pickles/text.pickle', 'wb')
    pickle.dump(text, file_text)

    file_label = open('pickles/label.pickle', 'wb')
    pickle.dump(label, file_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pickle Builder')

    parser.add_argument('--vocab_size', type=int, default=25000)

    config = parser.parse_args()

    build_tokenizer()
    build_vocab(config)
