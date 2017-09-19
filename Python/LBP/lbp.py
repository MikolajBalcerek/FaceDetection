
import cv2
from skimage.feature import local_binary_pattern
from load_faces import prepare_data
from scipy.stats import itemfreq
import numpy as np



def calc_lbps(X, method, radius):
    """Mikołaj oblicza lbps."""
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
    #przygotowanie danych
    Xtrain, Ytrain = prepare_data('train')
    Xtest, Ytest = prepare_data('test')

    # testowanie różnych opcji masowo
    # parametry dla calc_lbps(X)
    options = ["ror", "uniform", "var", "nri_uniform"];
    radius_range = 5;
    multiply_list = [1, 2, 4, 6, 8, 16];



    mass_results = []; #tablica z wynikiem wszystkich testów
    for option in options:
        for r in range(0,radius_range):
            print("Test dla opcji     " + option + " " + "radius " + str(r) + "\n");
            train_lbps  = calc_lbps(Xtrain, option, r);
            train_histograms = calc_histograms(train_lbps);

            test_lbps = calc_lbps(Xtest, option, r);
            test_histograms = calc_histograms(test_lbps);

            for img in train_lbps:
                cv2.imshow("TRAIN LBPS", img)
                cv2.waitKey(20);

            for img in test_lbps:
                cv2.imshow("TEST LBPS", img)
                cv2.waitKey(80);

            results = np.array([classify_histogram(hist, train_histograms)[0] for hist in test_histograms]);
            #print("Accuracy to {}".format(calc_acc(results, Ytest)))

            #dodanie do massresults tablicy;

            mass_results.append([option, r, format(calc_acc(results, Ytest))]);

    #wypisanie wyjścia
    print("Wyjście dla wszystkich opcji: \n")
    for output in mass_results:
        print(str(output[0]) + " | " + str(output[1]) + " | " + str(output[2]));
