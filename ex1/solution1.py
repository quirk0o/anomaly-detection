import numpy as np


def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    mean = np.mean(train_data)
    std_dev = np.std(train_data)
    thresh_low = mean - 3 * std_dev
    thresh_high = mean + 3 * std_dev

    print("Mean={}\nStandard deviation={}\nLow={}\nHigh={}".format(
        mean,
        std_dev,
        thresh_low,
        thresh_high
    ))

    results = [1 if x < thresh_low or x > thresh_high else 0 for x in test_data]
    return results
