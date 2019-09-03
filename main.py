import torch
import argparse

from trainer import Trainer
from utils import load_dataset, make_iter


def main(config, local):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, valid_data, test_data = load_dataset(config.random_seed)
    train_iter, valid_iter, test_iter = make_iter(train_data, valid_data, test_data, config.batch_size,
                                                  config.vocab_size, device)

    trainer = Trainer(config, train_iter, valid_iter, test_iter)

    if config.mode == 'train':
        trainer.train()

    elif config.mode == 'test':
        trainer.inference()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentiment Analysis')

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epoch', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=32)
    parser.add_argument('--max_seq_length', type=int, default=256)

    # Model Hyper-parameters
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--vocab_size', type=int, default=25000)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)

    # Additional options
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--save_model', type=str, default='model.pt')

    config = parser.parse_args()

    main(config, local=locals())
