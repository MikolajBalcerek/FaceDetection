
import cv2
from skimage.feature import local_binary_pattern
from load_faces import prepare_data
from scipy.stats import itemfreq
import numpy as np
from datetime import datetime


def calc_lbps(X, method, radius, size):
    """Mikołaj oblicza lbps."""
    # inne opcje to ror, uniform, var, nri_uniform
    return np.array([local_binary_pattern(cv2.resize(img, (size, size)), radius*8, radius, method=method) for img in X])
    # return np.array([local_binary_pattern(img, radius * 8, radius, method=method) for img in X])


def show_lbps(X):
    lbps = calc_lbps(X)
    for img in lbps:
        cv2.imshow("ale super", img)
        cv2.waitKey(80000)


def calc_hists(lbps):
    return np.array([itemfreq(x.reshape(-1)) for x in lbps])


def find_subarray(x, y):
    indices = np.argsort(x)
    sorted_x = x[indices]
    sorted_indices = np.searchsorted(sorted_x, y)
    return np.take(indices, sorted_indices, mode="clip")


def fit_hists_fast(hist1, hist2):
    sum = np.union1d(hist1[:, 0], hist2[:, 0])
    res1, res2 = np.zeros(sum.shape[0]), np.zeros(sum.shape[0])
    indexes1 = find_subarray(sum, hist1[:, 0])
    indexes2 = find_subarray(sum, hist2[:, 0])
    res1[indexes1] = hist1[:, 1]
    res2[indexes2] = hist2[:, 1]
    return res1 / np.max(res1), res2 / np.max(res2)


def fit_hists_slow(hist1, hist2):
    for x in hist1[:, 0]:
        if x not in hist2[:, 0]:
            hist2 = np.vstack((hist2, np.array([x, 0])))
    for x in hist2[:, 0]:
        if x not in hist1[:, 0]:
            hist1 = np.vstack((hist1, np.array([x, 0])))
    return hist1[:, 1] / max(hist1[:, 1]), hist2[:, 1]/ max(hist2[:, 1])


def classify_histogram(hist, train_hists, mthd="chisqr", spd="slow"):
    scores = []
    method = {
        "chisqr": cv2.HISTCMP_CHISQR,
        "chisqr_alt": cv2.HISTCMP_CHISQR_ALT,
        "correl": cv2.HISTCMP_CORREL,
        "hellinger": cv2.HISTCMP_HELLINGER,
        "kl_div": cv2.HISTCMP_KL_DIV,
        "intersect": cv2.HISTCMP_INTERSECT
    }[mthd]
    for th in train_hists:
        histNormal, thNormal = fit_hists_fast(hist, th) if spd == "fast" else fit_hists_slow(hist, th)
        scores.append(cv2.compareHist(histNormal.astype(np.float32), thNormal.astype(np.float32), method))
    scores = np.array(scores)
    index = np.argmax(scores) if mthd in ['correl', 'intersect', 'kl_div'] else np.argmin(scores)
    return Ytrain[index], scores[index]


def calc_acc(results, Ytest):
    return np.sum(results == Ytest) / Ytest.shape[0]


def calc_all(Xtrain, option, r, method, speed, size, show=False):
    train_lbps = calc_lbps(Xtrain, option, r,size)
    train_histograms = calc_hists(train_lbps)

    test_lbps = calc_lbps(Xtest, option, r, size)
    test_histograms = calc_hists(test_lbps)

    if show:
        # WYŚWIETLANIE obrazków
        for img in train_lbps:
            cv2.imshow("TRAIN LBPS", img)
            cv2.waitKey(10)

        for img in test_lbps:
            cv2.imshow("TEST LBPS", img)
            cv2.waitKey(10)

    results = np.array([classify_histogram(hist, train_histograms, spd=speed, mthd=method)[0]
                        for hist in test_histograms])
    print("Results to {}".format(results))
    print("Accuracy to {}".format(calc_acc(results, Ytest)))
    print("DONE")

    # zwrócenie wyniku
    return [option, r, format(calc_acc(results, Ytest))]


if __name__ == "__main__":
    #przygotowanie danych
    Xtrain, Ytrain = prepare_data('train')
    Xtest, Ytest = prepare_data('test')

    # testowanie różnych opcji masowo
    # parametry dla calc_lbps(X)


    #TUTAJ MOŻNA ZMIENIAĆ PARAMETRY

    radius_max_range = 2; #maksymalna wartość dla radius którą osiągnie w testowaniu
    radius_min_range = 1; # minimalna wartość dla radius od której się zacznie sprawdzanie możliwości
    radius_increment = 1; #inkrement po którym ma się zmieniać radius
    multiply_list = [1, 2, 4, 6, 8, 16];
    listSizes = [50, 100, 255, 500];
    # var produkuje zbyt duże liczby, by na razie algorytm miał sensowną złożoność
    options = ["default", "ror", "uniform", "nri_uniform"]
    methods = ["chisqr", "chisqr_alt", "correl", "hellinger", "kl_div", "intersect"]
    # metody porównywania hustogramów
    speeds = ['fast'] #, 'slow']
    #options = ["default"];
    multiply_list = [1, 2, 4, 6, 8, 16]



    #zapisywanie do pliku wyników na żywo
    #nazwa pliku
    nazwa = "WYNIK_" + str(datetime.now().timestamp()) + ".txt";
    #otwórz lub stwórz plik
    file = open(nazwa, 'w+')
    file.write("lbp option   |  radius |  result   |   method comparing histograms  |  speed  \n");
    file.close()

    mass_results = [] #tablica z wynikiem wszystkich testów

    for speed in speeds:
        for size in listSizes:
            for method in methods:
                for option in options:
                    for r in range(radius_min_range, radius_max_range, radius_increment):
                        print("Obliczenia dla speed = {0}, method = {1}, option = {2}, radius = {3}, size = {4} ".format(
                        speed, method, option, r, size))
                        mass_results.append(calc_all(Xtrain, option, r, method, speed, size));

                            #Dopisywanie do pliku
                        file = open(nazwa, 'a')
                        file.write(str(mass_results[-1]) + "  " + method + " " + speed + " " + str(size) + "\n");
                        file.flush()
                        file.close()



    #wypisanie wyjścia
    print("Wyjście dla wszystkich opcji: \n")
    print("method lbp option   |  radius |  result   |   method comparing histograms  |  speed   |  size \n")
    for output in mass_results:
        print(str(output[0]) + " | " + str(output[1]) + " | " + str(output[2]) + str(output[3]) + str(output[4]) + str(output[5]));