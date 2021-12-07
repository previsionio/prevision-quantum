import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from prevision_quantum_nn.dataset.dataset import DataSet
import prevision_quantum_nn as qnn


def get_gaussians_dataset(num_samples=50, holdout=20):
    """
    num_samples, default 50
    holdout in percentage, default 20%
    """
    scaler = MinMaxScaler((0., 1.))
    scaler.fit(X=np.array([[-10, -10], [10, 10]]))
    samples = scaler.transform(
        np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], num_samples))
    samples2 = scaler.transform(
        np.random.multivariate_normal([-5, -5], [[1, 0], [0, 1]], num_samples))
    samples3 = scaler.transform(
        np.random.multivariate_normal([-5, 5], [[1, 0], [0, 1]], num_samples))
    samples4 = scaler.transform(
        np.random.multivariate_normal([5, -5], [[1, 0], [0, 1]], num_samples))
    y_sample = np.zeros((num_samples, 1))
    y_sample2 = np.zeros((num_samples, 1))
    y_sample3 = np.ones((num_samples, 1))
    y_sample4 = np.ones((num_samples, 1))

    x = np.vstack((samples, samples2, samples3, samples4))
    y = np.vstack((y_sample, y_sample2, y_sample3, y_sample4)).flatten()
    samplesA = np.vstack((samples, samples2))
    samplesB = np.vstack((samples3, samples4))
    x, y = shuffle(x, y)
    num_samples = len(x)

    split_id = int(holdout * num_samples / 100)
    x_test = x[:split_id]
    y_test = y[:split_id]
    x = x[split_id:]
    y = y[split_id:]
    return x, y, x_test, y_test


if __name__ == "__main__":
    # get dataset
    x_train, y_train, x_val, y_val = get_gaussians_dataset()
    dataset = DataSet().from_numpy(x_train, y_train, val_features=x_val,
                                   val_labels=y_val)

    application = qnn.get_application("classification")
    application.solve(dataset)
