
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
    return np.array([hist[:, 1]/sum(hist[:, 1]) for hist in histograms])
    # return np.vectorize(lambda hist: hist[:, 1]/sum(hist[:, 1]))(histograms)


def classify_histogram(hist, train_hists):
    hist = np.array(hist, dtype=np.float32)
    train_hists = np.array([np.array(x, dtype=np.float32) for x in train_hists])
    scores = np.array([cv2.compareHist(hist, x, cv2.HISTCMP_CORREL) for x in train_hists])
    print("Score to {}".format(scores))
    index = np.argmin(scores)
    return Ytrain[index], scores[index]


def calc_acc(results, Ytest):
    return np.sum(results == Ytest) / Ytest.shape[0]


if __name__ == "__main__":
    Xtrain, Ytrain = prepare_data('train')
    Xtest, Ytest = prepare_data('test')

    train_lbps = calc_lbps(Xtrain)
    train_histograms = calc_histograms(train_lbps)

    test_lbps = calc_lbps(Xtest)
    test_histograms = calc_histograms(test_lbps)

    results = np.array([classify_histogram(hist, train_histograms)[0] for hist in test_histograms])
    print("Accuracy to {}".format(calc_acc(results, Ytest)))