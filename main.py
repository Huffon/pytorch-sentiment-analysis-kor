import argparse

from trainer import Trainer
from utils import load_dataset, make_iter


def main(config, local):
    if config.mode == 'train':
        train_data, valid_data = load_dataset(config.mode, config.random_seed)
        train_iter, valid_iter, pad_idx = make_iter(config.batch_size, config.mode, train_data=train_data,
                                                    valid_data=valid_data)

        trainer = Trainer(config, pad_idx, train_iter=train_iter, valid_iter=valid_iter)

        trainer.train()

    elif config.mode == 'test':
        test_data = load_dataset(config.mode, config.random_seed)
        test_iter, pad_idx = make_iter(config.batch_size, config.mode, test_data=test_data)
        trainer = Trainer(config, pad_idx, test_iter=test_iter)

        trainer.inference()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentiment Analysis')

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epoch', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=32)

    # Model Hyper-parameters
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--vocab_size', type=int, default=25000)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--bidirectional', type=bool, default=True)

    # Additional options
    parser.add_argument('--model', type=str, default='vanilla_rnn', choices=['vanilla_rnn', 'bidirectional_lstm'])
    parser.add_argument('--optim', type=str, default='Adam', choices=['SGD', 'Adam'])
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--save_model', type=str, default='model.pt')

    config = parser.parse_args()

    main(config, local=locals())
