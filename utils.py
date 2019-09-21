import os
import torch
import pickle
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

from torchtext import data as ttd
from torchtext.data import Example
from torchtext.data import Dataset
from soynlp.tokenizer import LTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_data(data):
    """
    Pre-process input DataFrame to have string label and not to have 'id' column
    Args:
        data: (DataFrame) input DataFrame which will be pre-processed

    Returns:

    """
    # remove 'id' column since it is not used to build torchtext Dataset
    data = data.iloc[:, 1:]

    # convert integer label (0 / 1) to string label ('neg' / 'pos')
    data.loc[data['label'] == 0, ['label']] = 'neg'
    data.loc[data['label'] == 1, ['label']] = 'pos'

    # drop missing values from input DataFrame
    missing_rows = []
    for idx, row in data.iterrows():
        if type(row.document) != str:
            missing_rows.append(idx)
    data = data.drop(missing_rows)

    return data


def load_dataset(mode, seed):
    """
    Load train and test dataset and split train dataset to make validation dataset.
    And finally convert train, validation and test dataset to pandas DataFrame.
        mode: (string) configuration mode used to which dataset to load
        seed: (integer) used to equally split train dataset when 'train_test_split' is called
    Args:

    Returns:
        (DataFrame) train, valid, test dataset converted to pandas DataFrame
    """
    print(f'Load NSMC dataset and convert it to pandas DataFrame . . .')

    data_dir = Path().cwd() / 'data'

    if mode == 'train':
        train_txt = os.path.join(data_dir, 'train.txt')
        train_data = pd.read_csv(train_txt, sep='\t')
        train_data, valid_data = train_test_split(train_data, test_size=0.3, random_state=seed)

        train_data = preprocess_data(train_data)
        valid_data = preprocess_data(valid_data)

        print(f'Number of training examples: {len(train_data)}')
        print(f'Number of validation examples: {len(valid_data)}')

        return train_data, valid_data

    else:
        test_txt = os.path.join(data_dir, 'test.txt')
        test_data = pd.read_csv(test_txt, sep='\t')

        test_data = preprocess_data(test_data)

        print(f'Number of testing examples: {len(test_data)}')

        return test_data


def convert_to_dataset(data, text, label):
    """
    Pre-process input DataFrame and convert pandas DataFrame to torchtext Dataset.
    Args:
        data: (DataFrame) pandas DataFrame to be converted into torchtext Dataset
        text: torchtext Field containing sentence information
        label: torchtext Field containing label information

    Returns:
        (Dataset) torchtext Dataset
    """
    # convert each row of DataFrame to torchtext 'Example' which contains text and label Fields
    list_of_examples = [Example.fromlist(row.tolist(),
                                         fields=[('text', text), ('label', label)]) for _, row in data.iterrows()]

    # construct torchtext 'Dataset' using torchtext 'Example' list
    dataset = Dataset(examples=list_of_examples, fields=[('text', text), ('label', label)])

    return dataset


def make_iter(batch_size, mode, train_data=None, valid_data=None, test_data=None):
    """
    Convert pandas DataFrame to torchtext Dataset and build iterator used to both train and test modess
    Args:
        batch_size: (integer) batch size used to make iterators
        mode: (string) configuration mode used to which iterator to make
        train_data: (DataFrame) pandas DataFrame used to build train iterator
        valid_data: (DataFrame) pandas DataFrame used to build validation iterator
        test_data: (DataFrame) pandas DataFrame used to build test iterator

    Returns:
        (BucketIterator) train, valid, test iterator
    """
    # load text and label field made by build_pickles.py
    file_text = open('pickles/text.pickle', 'rb')
    text = pickle.load(file_text)
    pad_idx = text.vocab.stoi[text.pad_token]

    file_label = open('pickles/label.pickle', 'rb')
    label = pickle.load(file_label)

    # convert pandas DataFrame to torchtext dataset
    if mode == 'train':
        train_data = convert_to_dataset(train_data, text, label)
        valid_data = convert_to_dataset(valid_data, text, label)

        # make iterator using train and validation dataset
        print(f'Make Iterators for training . . .')
        train_iter, valid_iter = ttd.BucketIterator.splits(
            (train_data, valid_data),
            # the BucketIterator needs to be told what function it should use to group the data.
            # In our case, we sort dataset using text of example
            sort_key=lambda sent: len(sent.text),
            # all of the tensors will be sorted by their length by below option
            sort_within_batch=True,
            batch_size=batch_size,
            device=device)

        return train_iter, valid_iter, pad_idx

    else:
        test_data = convert_to_dataset(test_data, text, label)

        # defines dummy list will be passed to the BucketIterator
        dummy = list()

        # make iterator using test dataset
        print(f'Make Iterators for testing . . .')
        test_iter, _ = ttd.BucketIterator.splits(
            (test_data, dummy),
            sort_key=lambda sent: len(sent.text),
            sort_within_batch=True,
            batch_size=batch_size,
            device=device)

        return test_iter, pad_idx


def pad_sentence(dataframe, min_len):
    """
    to use CNN, all the inputs has the minimum length as same as the largest filter size
    if the input is shorter than the largest CNN filter size, we should pad that input using pad_sentence method
    Args:
        dataframe: (DataFrame) dataframe used to train and validate the model
        min_len: (integer) the largest CNN filter size used to set the minimum length of the model

    Returns:

    """
    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores)

    for i, row in dataframe.iterrows():
        tokenized = tokenizer.tokenize(row.document)
        if len(tokenized) < min_len:
            tokenized += ['<pad>'] * (min_len - len(tokenized))
        padded_sent = ' '.join(tokenized)
        dataframe.at[i, 'document'] = padded_sent

    return dataframe


def binary_accuracy(predictions, targets):
    """
    Calculates binary accuracy for the model
    Args:
        predictions: predictions made by defined model
        targets: ground-truths

    Returns:
        (float) accuracy model made
    """
    # round predictions to the closest integer (0 or 1)
    rounded_preds = torch.round(torch.sigmoid(predictions))

    # convert into float for division
    correct = (rounded_preds == targets).float()

    # rounded_preds = [ 1   0   0   1   1   1   0   1   1   1]
    # targets       = [ 1   0   1   1   1   1   0   1   1   0]
    # correct       = [1.0 1.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0]

    acc = correct.sum() / len(correct)
    # acc           = 8.0 / 10 => 0.8
    return acc


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
