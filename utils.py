import os
import torch
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

from torchtext import data as td
from torchtext.data import Example
from torchtext.data import Dataset

TEXT = td.Field()
LABEL = td.LabelField(dtype=torch.float)


def load_dataset(seed):
    print(f'Load NSMC data and convert it to Dataframe . . .')
    torch.manual_seed(seed)

    data_dir = Path().cwd() / 'data'
    train_txt = os.path.join(data_dir, 'train.txt')
    test_txt = os.path.join(data_dir, 'test.txt')

    train_data = pd.read_csv(train_txt, sep='\t')
    test_data = pd.read_csv(test_txt, sep='\t')
    train_data, valid_data = train_test_split(train_data, test_size=0.3, random_state=seed)

    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')

    return train_data, valid_data, test_data


def convert_to_dataset(data):
    # remove id column
    data = data.iloc[:, 1:]

    data.loc[data['label'] == 0, ['label']] = 'neg'
    data.loc[data['label'] == 1, ['label']] = 'pos'

    missing_rows = []
    for idx, row in data.iterrows():
        if type(row.document) != str:
            missing_rows.append(idx)

    data = data.drop(missing_rows)

    list_of_examples = [Example.fromlist(row.tolist(),
                                         fields=[('text', TEXT), ('label', LABEL)]) for _, row in data.iterrows()]

    dataset = Dataset(examples=list_of_examples, fields=[('text', TEXT), ('label', LABEL)])

    return dataset


def build_vocab(vocab_size, data):
    print(f'\nBuild vocab . . .')
    TEXT.build_vocab(data, max_size=vocab_size)
    LABEL.build_vocab(data)

    print(f'Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}')
    print(f'Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}')

    print(TEXT.vocab.freqs.most_common(20))


def make_iter(train_data, valid_data, test_data, batch_size, vocab_size, device='cpu'):
    train_data = convert_to_dataset(train_data)
    valid_data = convert_to_dataset(valid_data)
    test_data = convert_to_dataset(test_data)

    build_vocab(vocab_size, train_data)

    print(f'\nMake Iterators . . .')
    train_iter, valid_iter, test_iter = td.BucketIterator.splits(
        (train_data, valid_data, test_data),
        sort_key=lambda x: len(x.text),
        # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=False,
        batch_size=batch_size,
        device=device)

    return train_iter, valid_iter, test_iter


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
