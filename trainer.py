import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from utils import epoch_time
from models.vanilla_rnn import RNN
from evaluation import binary_accuracy

logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, config, train_iter, valid_iter, test_iter):
        self.config = config

        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.test_iter = test_iter

        self.model = RNN(self.config)
        self.model.to(device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr)

        # BCEWithLogitsLoss carries out both the sigmoid and the binary cross entropy steps.
        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion.to(device)

    def train(self):
        print(f'The model has {self.model.count_parameters():,} trainable parameters')
        best_valid_loss = float('inf')

        self.model.train()

        for epoch in range(self.config.num_epoch):
            epoch_loss = 0
            epoch_acc = 0

            start_time = time.time()

            for batch in self.train_iter:
                # For each batch, first zero the gradients
                self.optimizer.zero_grad()

                # predictions = [batch size, 1] -> [batch size])
                predictions = self.model(batch.text).squeeze(1)

                loss = self.criterion(predictions, batch.label)
                acc = binary_accuracy(predictions, batch.label)

                loss.backward()
                self.optimizer.step()

                # .item() method is used to extract a scalar
                # from a tensor which only contains a single value.
                epoch_loss += loss.item()
                epoch_acc += acc.item()

            train_loss = epoch_loss / len(self.train_iter)
            train_acc = epoch_acc / len(self.train_iter)

            valid_loss, valid_acc = self.evaluate()

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), 'model.pt')

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\tVal. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')

    def evaluate(self):
        epoch_loss = 0
        epoch_acc = 0

        self.model.eval()

        with torch.no_grad():
            for batch in self.valid_iter:
                predictions = self.model(batch.text).squeeze(1)

                loss = self.criterion(predictions, batch.label)
                acc = binary_accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(self.valid_iter), epoch_acc / len(self.valid_iter)

    def inference(self):
        epoch_loss = 0
        epoch_acc = 0

        self.model.load_state_dict(torch.load(self.config.save_model))
        self.model.eval()

        with torch.no_grad():
            for batch in self.test_iter:
                predictions = self.model(batch.text).squeeze(1)

                loss = self.criterion(predictions, batch.label)
                acc = binary_accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        test_loss = epoch_loss / len(self.test_iter)
        test_acc = epoch_acc / len(self.test_iter)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
