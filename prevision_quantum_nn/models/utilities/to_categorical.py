""" to categorical module """

import numpy as np
import tensorflow as tf


def to_categorical(train_labels, val_labels):
    """ transforms labels to categorical labels for multiclassification

    Args:
        train_labels
        val_labels
    Returns:
        train_labels
        val_labels
    """
    if val_labels is not None:
        labels = np.hstack((train_labels, val_labels))
    else:
        labels = train_labels
    labels = tf.keras.utils.to_categorical(labels)
    train_labels = labels[:len(train_labels)]
    val_labels = labels[len(train_labels):]
    return train_labels, val_labels
