import keras
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import random

class Data_Preprocessing():
    def __init__(self, x_TrainingPath, y_TrainingPath, x_TestingPath, y_TestingPath ):
        self.x_train_path = x_TrainingPath
        self.y_train_path = y_TrainingPath
        self.x_test_path = x_TestingPath
        self.y_test_path = y_TestingPath

    def load_date(self):
        self.x_train_0 = np.loadtxt(self.x_train_path)
        self.y_train = np.loadtxt(self.y_train_path)
        self.x_test_0 = np.loadtxt(self.x_test_path)
        self.y_test = np.loadtxt(self.y_test_path)
        #return self.x_train, self.y_train, self.x_test, self.y_test

    #Transform the dataset
    def standardScale(self):
        sc = StandardScaler()
        self.x_train_1 = sc.fit_transform(self.x_train_0)
        self.x_test_1 = sc.transform(self.x_test_0)

    #PCA preprocessing for X_Train, X_Test to one dimension
    def PCA_preprocessing(self):
        pca = PCA.transform(1)
        self.x_train_2 = pca.fit_transform(self.x_train_1)
        self.x_test_2 = pca.fit_transform(self.x_test_1)
        return self.x_train_2, self.x_test_2, self.y_train, self.y_test

class Bayesian_Logistic_Regression():
    def __init__(self):
        self.GammaParameters = [1, 0.1]
        self.Gamma = random.gammavariate(self.GammaParameters[0], self.GammaParameters[1])
        self.MultiNormal_Cov = self.Gamma










