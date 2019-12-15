from sklearn.svm import OneClassSVM

from utils import binary2neg_boolean
import numpy as np


def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    svm = OneClassSVM(nu=0.1).fit(train_data)

    predict = svm.predict(test_data)

    result = binary2neg_boolean(predict)
    return result
