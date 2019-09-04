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
    """
    Train soynlp tokenizer which will be used to tokenize input sentence
    Returns:
        (tokenizer) soynlp tokenizer
        (train_txt) train text which will be used to build vocabulary
    """
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

    ltokenizer = LTokenizer(scores=cohesion_scores)

    return ltokenizer, train_txt


def build_vocab(config, tokenizer, train_txt):
    """
    Build vocab used to convert input sentence into word indices using soynlp tokenizer
    Args:
        config: configuration containing various options
        tokenizer: soynlp tokenizer used to struct torchtext Field object

    Returns:

    """
    # To use packed padded sequences, tell the model how long the actual sequences are
    text = ttd.Field(tokenize=tokenizer.tokenize, include_lengths=True)
    label = ttd.LabelField(dtype=torch.float)

    train_data = pd.read_csv(train_txt, sep='\t')
    train_data, valid_data = train_test_split(train_data, test_size=0.3, random_state=32)
    train_data = convert_to_dataset(train_data, text, label)

    print(f'Build vocabulary using torchtext . . .')
    text.build_vocab(train_data, max_size=config.vocab_size)
    label.build_vocab(train_data)

    print(f'Unique tokens in TEXT vocabulary: {len(text.vocab)}')
    print(f'Unique tokens in LABEL vocabulary: {len(label.vocab)}')

    print(f'Most commonly used words are as follows:')
    print(text.vocab.freqs.most_common(20))

    file_text = open('pickles/text.pickle', 'wb')
    pickle.dump(text, file_text, pickle.HIGHEST_PROTOCOL)

    file_label = open('pickles/label.pickle', 'wb')
    pickle.dump(label, file_label, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentiment Analysis')

    parser.add_argument('--vocab_size', type=int, default=25000)
    config = parser.parse_args()

    tokenizer, train_txt = build_tokenizer()
    build_vocab(config, tokenizer, train_txt)
