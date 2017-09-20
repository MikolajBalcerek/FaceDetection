import cv2
import os
import numpy as np


def load_data(purpose, num_people=5):
    """Wczytuje obrazki i przyporządkowane im etykiety. Działa pod Windą (\\ zamiast /).
    purpose - train / test, 
    num_people - liczba osób."""
    X, Y = [], []
    for i in range(num_people):
        current_dir = "Data\\" + purpose + "\\" + str(i)
        dirs = os.listdir(current_dir)
        print("Reading images at path " + current_dir + ':')
        for name in dirs:
            path = os.path.abspath(current_dir + '\\' + name)
            X.append(cv2.imread(path, 0))
            # 0 oznacza, że wczytujemy tylko czarno-białe zdjęcia, jak coś, to mogę to zmienić.
            Y.append(i)
            print("\t file " + path + " read.")
    return np.array(X), np.array(Y)


def detect_face(img, scaleFactorgiven, minNeighborsgiven):
    """Wykrywa twarz na obrazku img. Zwraca mniejszy obrazek, zawierający twarz."""
    # TODO CascadeClassifier i detect Multiscale ssą - nie wykrywają zawsze twarzy, sprawność na start to 9/16
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) - zdjęcia są szare na wejściu
    face_cascade = cv2.CascadeClassifier('cascade.xml')
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=7)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    # [0] - interesuje nas tylko pierwsza twarz na zdjęciu, domyślnie każde zdj powinno mieć tylko jedną twarz
    return img[y:y + w, x:x + h]


def prepare_data(purpose):
    """Przygotowuje dane. Odpowiedzialna za preprocessing i normalizacje.
        purpose - train / test"""
    X, Y = load_data(purpose)
    X = np.array([detect_face(img, 1.2, 7) for img in X])
    detected = [i for i, img in enumerate(X) if img is not None]
    return X[detected], Y[detected]


def show_undetected_faces(seconds=1):
    for abc in ("train", "test"):
        X, Y = load_data(abc)
        faces = np.array([detect_face(img) for img in X])
        undetected = [i for i, img in enumerate(faces) if img is None]
        for x in X[undetected]:
            cv2.imshow("Smuteczek", x)
            cv2.waitKey(seconds*1000)



def show_detected_faces(seconds=0.5):
    for abc in ("train", "test"):
        X, Y = load_data(abc)
        faces = np.array([detect_face(img) for img in X])
        detected = [i for i, img in enumerate(faces) if img is not None];
        for x in X[detected]:
            cv2.imshow("Znalazlem go", x);
            cv2.waitKey(1000);


def count_detection_tests(scaleFactor, minNeighbors):

    #liczenie wyników testów
    countDetectedAll = 0;
    countPhotosAll = 0;

    print("\n Rozpoczynam wykonanie testów: \n");
    for abc in ("train", "test"):
        print("Zestaw " + str(abc) + "\n");

        X, Y = load_data(abc)
        countPhotos = X.size;
        print("Załadowano " + str(countPhotos) + " zdjęć \n");

        print("Wykrywam twarze funkcją detect_face..\n")
        faces = np.array([detect_face(img, scaleFactor, minNeighbors) for img in X])
        detected = [i for i, img in enumerate(faces) if img is not None];
        countDetected = X[detected].size;
        print(str(abc) + ":  Znaleziono " + str(countDetected) + " z wszystkich " + str(countPhotos) + "\n");

        countPhotosAll = countPhotosAll + countPhotos;
        countDetectedAll = countDetectedAll + countDetected;

    print("WSZYSTKIE" + ":  Znaleziono " + str(countDetectedAll) + " z wszystkich " + str(countPhotosAll) + "\n");
    return [countDetectedAll, countPhotosAll];

if __name__ == '__main__':

    #PARAMETRY DO ZMIENIANIA DO TESTÓW
    #  tutaj można grzebać :)
    scaleFactor_min_range = 0; #minimalny zasięg parametru scaleFactor, wartość /100
    scaleFactor_max_range = 1000000; #maksymalny zasięg parametru scaleFactor, wartość /100
    scaleFactor_increment = 300000; #inkrement dla testów scaleFactor, wartość/100
    minNeighbors_min_range = 0;  # minimalny zasięc parametru minNeighbors
    minNeighbors_max_range = 1000000; #maksymalny zasięc parametru minNeighbors
    minNeighbors_increment = 300000; #inkrement dla testów minNeighbors

    results = []; #tablica zawierająca wyniki wszystkich testów
    #testowanie dla ręcznie dobranyc# h zmiennych
    score = count_detection_tests(1.2, 7);
    results.append([1.2, 7, [score]]);


    #badanie wpływu zmiennych na rozpoznawanie twarzy
    for scaleFactor in range (scaleFactor_min_range, scaleFactor_max_range, scaleFactor_increment):
        for minNeighbors in range (minNeighbors_min_range, minNeighbors_max_range, minNeighbors_increment):
            scaleFactor = scaleFactor / 100; #inkrementacja po 0.1 * inkrment, nie da się for z float increments w Python 3
            score = count_detection_tests(scaleFactor, minNeighbors);
            results.append([scaleFactor, minNeighbors, [score]]);


    #wypisywanie rezultatów
    print("Wypisywanie wyników eksperymentu: \n")
    print("scaleFator | minNeighbors | Detected | Out of")
    for result in results:
        print(str(result[0]) + " | " + str(result[1]) + " | " + str(result[2]));


    #sprawdzanie czy ratio dla wykrywania zmieniło się kiedykolwiek dla wszystkich parametrów wobec ilości sukcesów pierwszego testu (dobranego przez Mietka)
    if ((not results[0][2]) in results):
        print("WYNIK ZMIENIŁ SIĘ W ZALEŻNOŚCI OD PARAMETRU!!!!");

#show_undetected_faces();
#show_detected_faces();

# ja = cv2.imread("szalone.jpg")
#     cv2.imshow("przed", ja)
#     cv2.waitKey(100)
#     face, rect = detect_face(ja)
#
#     Xtrain, Ytrain = prepare_data('train')
#     Xtest, Ytest = prepare_data('test')
#     for i, img in enumerate(Xtrain):
#         cv2.imshow('index of this person is ' + str(Ytrain[i]), img)
#         cv2.waitKey(4000)
