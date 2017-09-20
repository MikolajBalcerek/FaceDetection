
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


def calc_freqs(lbps):
    return np.array([itemfreq(x.reshape(-1)) for x in lbps])


def calc_histograms(hist1, hist2):
    print(hist1.shape)
    for x in hist1[:, 0]:
        if x not in hist2[:, 0]:
            hist2 = np.vstack((hist2, np.array([x, 0])))
    for x in hist2[:, 0]:
        if x not in hist1[:, 0]:
            hist1 = np.vstack((hist1, np.array([x, 0])))
    return hist1[:, 1]/max(hist1[:, 1]), hist2[:, 1]/max(hist2[:, 1])


def classify_histogram(hist, train_hists):
    scores = []
    for th in train_hists:
        histNormal, thNormal = calc_histograms(hist, th)
        scores.append(cv2.compareHist(np.array(histNormal, dtype=np.float32), np.array(thNormal, dtype=np.float32),
                                      cv2.HISTCMP_CORREL))
    scores = np.array(scores)
    print("Score to {}".format(scores))
    index = np.argmax(scores)
    return Ytrain[index], scores[index]


def calc_acc(results, Ytest):
    return np.sum(results == Ytest) / Ytest.shape[0]


if __name__ == "__main__":
    #przygotowanie danych
    Xtrain, Ytrain = prepare_data('train')
    Xtest, Ytest = prepare_data('test')

    # testowanie różnych opcji masowo
    # parametry dla calc_lbps(X)
    options = ["ror", "uniform", "var", "nri_uniform"]; #opcje testowania lbp

    #TUTAJ MOŻNA ZMIENIAĆ PARAMETRY

    radius_max_range = 3; #maksymalna wartość dla radius którą osiągnie w testowaniu
    radius_min_range = 1; # minimalna wartość dla radius od której się zacznie sprawdzanie możliwości
    radius_increment = 1; #inkrement po którym ma się zmieniać radius
    multiply_list = [1, 2, 4, 6, 8, 16];
    # var produkuje zbyt duże liczby, by na razie algorytm miał sensowną złożoność
    options = ["default", "ror", "uniform", "nri_uniform"]
    radius_range = 3
    multiply_list = [1, 2, 4, 6, 8, 16]
    #
    #
    #
    #



    mass_results = []; #tablica z wynikiem wszystkich testów
    for option in options:
        for r in range(radius_min_range, radius_max_range, radius_increment):
            print("Test dla opcji     " + option + " " + "radius " + str(r) + "\n");
            train_lbps = calc_lbps(Xtrain, option, r);
            train_histograms = calc_histograms(train_lbps);

            test_lbps = calc_lbps(Xtest, option, r)
            test_histograms = calc_freqs(test_lbps)

            # for img in train_lbps:
            #     cv2.imshow("TRAIN LBPS", img)
            #     cv2.waitKey(10)
            #
            # for img in test_lbps:
            #     cv2.imshow("TEST LBPS", img)
            #     cv2.waitKey(10)

            results = np.array([classify_histogram(hist, train_histograms)[0] for hist in test_histograms])
            print("Results to {}".format(results))
            print("Accuracy to {}".format(calc_acc(results, Ytest)))

            #dodanie do massresults tablicy;

            mass_results.append([option, r, format(calc_acc(results, Ytest))])



    #wypisanie wyjścia
    print("Wyjście dla wszystkich opcji: \n")
    print("method  |  radius |  result   \n")
    for output in mass_results:
        print(str(output[0]) + " | " + str(output[1]) + " | " + str(output[2]));