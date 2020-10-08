""" losses module """
import pennylane.numpy as np


def square_loss(labels, predictions):
    """Square loss.

    Args:
        labels (array[float]): 1-d array of labels
        predictions (array[float]): 1-d array of predictions

    Returns:
        float: square loss
    """
    loss = 0
    for label, pred in zip(labels, predictions):
        loss = loss + (label - pred) ** 2

    loss = loss / len(labels)
    return loss


def cross_entropy(labels, predictions):
    """ Categorical cross entropy loss function

    Args:
        labels (array[float]): 1-d array of labels
        predictions (array[float]): 1-d array of predictions

    Returns:
        float: cross entropy
    """
    loss = 0
    for label, pred in zip(labels, predictions):
        for label_, pred_ in zip(label, pred):
            loss -= label_ * np.log(pred_)
    loss = loss / len(labels)
    return loss
