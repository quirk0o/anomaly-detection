import numpy as np
from sklearn.covariance import MinCovDet


def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    est_conv = MinCovDet().fit(train_data)

    print(est_conv.covariance_)

    train_dist = np.abs(est_conv.mahalanobis(train_data))
    max_train_dist = np.max(train_dist)
    test_dist = np.abs(est_conv.mahalanobis(test_data))

    print("Max train dist={}".format(max_train_dist))

    result = [1 if x > max_train_dist else 0 for x in test_dist]
    return result
