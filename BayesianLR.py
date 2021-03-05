import keras
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import random

class Data_Preprocessing():
    def __init__(self, x_TrainingPath, x_TestingPath, y_TrainingPath, y_TestingPath ):
        self.x_train_path = x_TrainingPath
        self.y_train_path = y_TrainingPath
        self.x_test_path = x_TestingPath
        self.y_test_path = y_TestingPath

    def load_data(self):
        self.x_train_0 = np.loadtxt(self.x_train_path)
        self.y_train = np.loadtxt(self.y_train_path)
        self.x_test_0 = np.loadtxt(self.x_test_path)
        self.y_test = np.loadtxt(self.y_test_path)
        self.y_train, self.y_test = self.label_process(self.y_train, self.y_test)

    def label_process(self, y_train, y_test):
        trainlabels = list()
        for i in y_train:
            i = int(i)
            if i == 1:
                i = 0
            if i == 2:
                i = 1
            trainlabels.append(int(i))
        y_train = trainlabels

        testlabels = list()
        for i in y_test:
            i = int(i)
            if i == 1:
                i = 0
            if i == 2:
                i = 1
            testlabels.append(int(i))
        y_test = testlabels
        return y_train, y_test

    #Transform the dataset
    def standardScale(self):
        sc = StandardScaler()
        self.x_train_1 = sc.fit_transform(self.x_train_0)
        self.x_test_1 = sc.transform(self.x_test_0)

    #PCA preprocessing for X_Train, X_Test to one dimension
    def PCA_preprocessing(self):
        pca = PCA(n_components = 1)
        self.x_train_2 = pca.fit_transform(self.x_train_1)
        self.x_test_2 = pca.fit_transform(self.x_test_1)
        return self.x_train_2, self.x_test_2

    #Generate X matrix and combine matrix together
    def GenerateX(self):
        # This function is used to generate X matrix, combining all features together
        # Both X_train and X_test are generated by this function
        # Should return a Matrix
        self.load_data()
        self.standardScale()
        return self.PCA_preprocessing()




class Bayesian_Logistic_Regression_Parameters():
    def __init__(self):
        self.GammaParameters = [1, 0.1]

        # initialize Gamma
        self.Gamma = random.gammavariate(self.GammaParameters[0], self.GammaParameters[1])
        # initialize theta = log beta
        self.MultiNormal_Cov = np.diag([self.Gamma] * 5)
        self.mean = (0, 0, 0, 0, 0)
        # initialize Beta
        self.beta = np.random.multivariate_normal(self.mean, self.MultiNormal_Cov, size=None, check_valid= 'raise')
        # X and Y are initialized and ready to use

    def getParameters(self):
        # check = np.dot(np.transpose(self.beta), self.test)
        return self.Gamma, self.MultiNormal_Cov, self.beta

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def Loss_function(Y_train, X_train, theta, beta, covariance):
    #This function is designed to calculate the loss function, which will
    #be used to determine when and where the Gradient Descent is going to end
    #Should return a float number
    Y_train = Y_train.reshape(1, 1032)
    cal1 = np.dot(X_train, beta)
    cal2 = np.log(np.ones((1, 1032)) + np.exp(-cal1))
    cal3 = np.log(np.ones((1, 1032)) + np.exp(cal1))
    cal4 = np.dot(Y_train, np.transpose(cal2))
    cal5 = np.dot(np.ones((1,1032))-Y_train, np.transpose(cal3))
    likelihood = -(cal4 + cal5)

    #cal6 = p * theta
    cal7 = np.exp(theta) * np.dot(beta, np.transpose(beta)) / 2
    #cal8 = (a - 1) * theta
    #cal9 = b * np.exp(theta)
    #posteria = cal6 - cal7 + cal8 - cal9
    #loss = likelihood + posteria

    return None

def Beta_update(x, y):
    # This function is going to update Beta when doing Gradient Descent
    # Should return a float number
    # To be Finished
    return None

def Theta_update(x, y):
    # This function is going to update Theta which Equals to log Gamma when doing Gradient Descent
    # Should return a float number
    # To be Finished
    return None








def run():
    # This function is used to
    # 1. Process matrix and 2. run gradient Descent

    #Part 1: preprocess the data and generate the matrix.

    #DFS and 2_Gaussian have smaller dataset
    Features = ['PCA', 'Gaussian', 'Gabor', 'Wavelet', 'STFT']
    Instance = list()
    X_train_lst = list()
    X_test_lst = list()

    for i in range(len(Features)):
        Instance.append(Data_Preprocessing('./TXT/'+Features[i]+'/x_train', './TXT/'+Features[i]+'/x_test', './TXT/'+Features[i]+'/y_train', './TXT/'+Features[i]+'/y_test'))
        train, test = Instance[i].GenerateX()
        X_train_lst.append(train)
        X_test_lst.append(test)

    X_train= np.hstack((X_train_lst[0], X_train_lst[1],X_train_lst[2], X_train_lst[3], X_train_lst[4]))
    X_test = np.hstack((X_test_lst[0], X_test_lst[1],X_test_lst[2], X_test_lst[3], X_test_lst[4]))
    Y_train = np.asarray(Instance[0].y_train)
    Y_test = np.asarray(Instance[1].y_test)

    #Part 2: run gradient Descent
    #Part 2.1: Generate Parameters
    Parameter = Bayesian_Logistic_Regression_Parameters()
    Gamma, covariance, beta = Parameter.getParameters()
    theta = np.log(Gamma)
    Loss_function(Y_train, X_train, theta, beta, covariance)

    res = np.dot(X_train, beta)



    return None

if __name__ == '__main__':
    run()
    print("Testing")










