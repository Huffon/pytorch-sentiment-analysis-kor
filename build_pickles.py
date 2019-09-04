import os
import pickle
import argparse
from pathlib import Path

import torch
import pandas as pd

from utils import convert_to_dataset
from torchtext import data as ttd
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from sklearn.model_selection import train_test_split


def build_tokenizer():
    print(f'Now building soy-nlp tokenizer . . .')
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


def build_vocab(config, local):
    pickle_in = open("pickles/tokenizer.pickle", "rb")
    cohesion_scores = pickle.load(pickle_in)
    tokenizer = LTokenizer(scores=cohesion_scores)

    # To use packed padded sequences, tell the model how long the actual sequences are
    TEXT = ttd.Field(tokenize=tokenizer.tokenize, include_lengths=True)
    LABEL = ttd.LabelField(dtype=torch.float)

    data_dir = Path().cwd() / 'data'
    train_txt = os.path.join(data_dir, 'train.txt')
    train_data = pd.read_csv(train_txt, sep='\t')
    train_data, valid_data = train_test_split(train_data, test_size=0.3, random_state=32)
    train_data = convert_to_dataset(train_data, TEXT, LABEL)

    print(f'\nBuild vocab . . .')
    TEXT.build_vocab(train_data, max_size=config.vocab_size)
    LABEL.build_vocab(train_data)

    print(f'Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}')
    print(f'Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}')

    print(TEXT.vocab.freqs.most_common(20))

    file_text = open('pickles/text.pickle', 'wb')
    pickle.dump(TEXT, file_text, pickle.HIGHEST_PROTOCOL)

    file_label = open('pickles/label.pickle', 'wb')
    pickle.dump(LABEL, file_label, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentiment Analysis')

    parser.add_argument('--vocab_size', type=int, default=25000)
    config = parser.parse_args()

    build_tokenizer()
    build_vocab(config, local=locals())
