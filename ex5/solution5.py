import numpy as np


def reconstruction_errors(inputs: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    """Calculate reconstruction errors.

    :param inputs: Numpy array of input images
    :param reconstructions: Numpy array of reconstructions
    :return: Numpy array (1D) of reconstruction errors for each pair of input and its reconstruction
    """
    errors_2d = np.asarray([np.square(reconstruction - input) for input, reconstruction in zip(inputs, reconstructions)])
    return np.mean(errors_2d, axis=1)


def calc_threshold(reconstr_err_nominal: np.ndarray) -> float:
    """Calculate threshold for anomaly-detection

    :param reconstr_err_nominal: Numpy array of reconstruction errors for examples drawn from nominal class.
    :return: Anomaly-detection threshold
    """
    mean = np.mean(reconstr_err_nominal)
    std_dev = np.std(reconstr_err_nominal)
    thresh = mean + 3 * std_dev
    return thresh


def detect(reconstr_err_all: np.ndarray, threshold: float) -> list:
    """Recognize anomalies using given reconstruction errors and threshold.

    :param reconstr_err_all: Numpy array of reconstruction errors.
    :param threshold: Anomaly-detection threshold
    :return: list of 0/1 values
    """
    results = [1 if x > threshold else 0 for x in reconstr_err_all]

    return results