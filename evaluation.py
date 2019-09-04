import torch


def binary_accuracy(predictions, targets):
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
