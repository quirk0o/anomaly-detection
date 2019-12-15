from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from utils import binary2neg_boolean
import numpy as np

SEED = 1


def detect_cov(data: np.ndarray, outliers_fraction: float) -> list:
    elle = EllipticEnvelope(contamination=outliers_fraction).fit(data)

    predict = elle.predict(data)

    result = binary2neg_boolean(predict)
    return result


def detect_ocsvm(data: np.ndarray, outliers_fraction: float) -> list:
    svm = OneClassSVM(nu=outliers_fraction).fit(data)

    predict = svm.predict(data)

    result = binary2neg_boolean(predict)
    return result


def detect_iforest(data: np.ndarray, outliers_fraction: float) -> list:
    forest = IsolationForest(contamination=outliers_fraction).fit(data)

    predict = forest.predict(data)

    result = binary2neg_boolean(predict)
    return result


def detect_lof(data: np.ndarray, outliers_fraction: float) -> list:
    predict = LocalOutlierFactor(contamination=outliers_fraction, n_neighbors=500).fit_predict(data)

    result = binary2neg_boolean(predict)
    return result
