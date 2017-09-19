# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np


def load_data(purpose, num_people=2):
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


def detect_face(img):
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
    X = np.array([detect_face(img) for img in X])
    detected = [i for i, img in enumerate(X) if img is not None]
    return X[detected], Y[detected]


def show_undetected_faces(seconds=20):
    for abc in ("train", "test"):
        X, Y = load_data(abc)
        faces = np.array([detect_face(img) for img in X])
        undetected = [i for i, img in enumerate(faces) if img is None]
        for x in X[undetected]:
            cv2.imshow("Smuteczek", x)
        aitKey(100)
# face, rect = detect_face(ja)

# Xtrain, Ytrain = prepare_data('train')
# Xtest, Ytest = prepare_data('test')
# for i, img in enumerate(Xtrain):
#     cv2.imshow('index of this person is ' + str(Ytrain[i]), img)
#     cv2.waitKey(4000)
