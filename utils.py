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
    """
    Load train and test dataset and split train dataset to make validation dataset.
    And finally convert train, validation and test dataset to pandas DataFrame.
    Args:
        seed: (integer) used to equally split train dataset when 'train_test_split' is called

    Returns:
        (DataFrame) train, valid, test dataset converted to pandas DataFrame
    """
    print(f'Load NSMC data and convert it to DataFrame . . .')

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
    """
    Pre-process input DataFrame and convert pandas DataFrame to torchtext Dataset.
    Args:
        data: (DataFrame)

    Returns:
        (Dataset) torchtext Dataset
    """
    # remove id column since it is not used to make torchtext Dataset
    data = data.iloc[:, 1:]

    # convert integer label (0 / 1) to string label ('neg' / 'pos')
    data.loc[data['label'] == 0, ['label']] = 'neg'
    data.loc[data['label'] == 1, ['label']] = 'pos'

    # drop some missing values
    missing_rows = []
    for idx, row in data.iterrows():
        if type(row.document) != str:
            missing_rows.append(idx)
    data = data.drop(missing_rows)

    # convert each row of DataFrame to torchtext 'Example' which contains text and label attributes
    list_of_examples = [Example.fromlist(row.tolist(),
                                         fields=[('text', TEXT), ('label', LABEL)]) for _, row in data.iterrows()]

    # make torchtext 'Dataset' using torchtext 'Example' list
    dataset = Dataset(examples=list_of_examples, fields=[('text', TEXT), ('label', LABEL)])

    return dataset


def build_vocab(data, vocab_size):
    """
    Build vocabulary used to convert input string to word index
    Args:
        data: (Dataset) torchtext 'train' Dataset used to build vocabulary
        vocab_size: (integer) the size of vocabulary of your model
    """
    print(f'\nBuild vocab . . .')
    TEXT.build_vocab(data, max_size=vocab_size)
    LABEL.build_vocab(data)

    print(f'Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}')
    print(f'Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}')

    print(TEXT.vocab.freqs.most_common(20))


def make_iter(train_data, valid_data, test_data, batch_size, vocab_size, device='cpu'):
    """
    Convert pandas DataFrame to torchtext Dataset and make iterator used to train and test
    Args:
        train_data: (DataFrame) pandas DataFrame used to make train iterator
        valid_data: (DataFrame) pandas DataFrame used to make validation iterator
        test_data: (DataFrame) pandas DataFrame used to make test iterator
        batch_size: (integer) batch size used to make iterators
        vocab_size: (integer) vocabulary size used to build vocabulary
        device: (string) device name where train and test is run on

    Returns:
        (BucketIterator) train, valid, test iterator
    """
    # convert pandas DataFrame to torchtext dataset
    train_data = convert_to_dataset(train_data)
    valid_data = convert_to_dataset(valid_data)
    test_data = convert_to_dataset(test_data)

    # build vocab using train dataset
    build_vocab(train_data, vocab_size)

    # make iterator using train, validation and test
    print(f'\nMake Iterators . . .')
    train_iter, valid_iter, test_iter = td.BucketIterator.splits(
        (train_data, valid_data, test_data),
        sort_key=lambda x: len(x.text),
        # the BucketIterator needs to be told what function it should use to group the data.
        # In our case, we sort dataset using text of example
        sort_within_batch=False,
        batch_size=batch_size,
        device=device)

    return train_iter, valid_iter, test_iter


def epoch_time(start_time, end_time):
    """
    Calculate the time spent to train one epoch
    Args:
        start_time: (float) training start time
        end_time: (float) training end time

    Returns:
        (int) elapsed_mins and elapsed_sec spent for one epoch

    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
