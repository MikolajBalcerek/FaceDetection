
import cv2
from skimage.feature import local_binary_pattern
from load_faces import prepare_data
from scipy.stats import itemfreq
import numpy as np


def calc_lbps(X):
    """Mikołaj oblicza lbps."""
    radius = 3
    method = 'default'
    # inne opcje to ror, uniform, var, nri_uniform
    return np.array([local_binary_pattern(img, radius*8, radius, method=method) for img in X])


def show_lbps(X):
    lbps = calc_lbps(X)
    for img in lbps:
        cv2.imshow("ale super", img)
        cv2.waitKey(80000)


def calc_histograms(lbps):
    histograms = np.array([itemfreq(lbp.ravel) for lbp in lbps])
    return np.vectorize(lambda hist: hist[:, 1]/sum(hist[:, 1]))(histograms)


def compare_histograms():
    # score = cv2.compareHist(np.array(x), np.array(hist), cv2.cv.CV_COMP_CHISQR)
    pass

if __name__ == "__main__":
    Xtrain, Ytrain = prepare_data('train')
    show_lbps(Xtrain)
    Xtest, Ytest = prepare_data('test')
    show_lbps(Xtest)