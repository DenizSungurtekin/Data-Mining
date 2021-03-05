import numpy as np
from random import shuffle
from classifier import Classifier


class Logistic(Classifier):
    """A subclass of Classifier that uses the logistic function to classify."""
    def __init__(self, random_seed=0):
        super().__init__('logistic')
        if random_seed:
            np.random.seed(random_seed)

    def sigmoide(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoideDerivative(self,x):
        return self.sigmoide(x)*(1-self.sigmoide(x))

    def loss(self, X, y=None, reg=0, newton = True):


        scores = []
        loss = None
        dW = np.zeros_like(self.W)
        ddW = np.zeros((3073, 3073))
        num_features = self.W.shape[0]
        num_train = X.shape[0]
        scores = [self.sigmoide(np.dot(np.transpose(self.W), np.reshape(X[i], (3073, 1)))) for i in range(num_train)]
        if y is None:
            return scores

        FirstTerm = [y[i]*np.log(self.sigmoide(np.dot(np.transpose(self.W), np.reshape(X[i],(3073,1))))) for i in range(num_train)]
        SecondTerm = [(1-y[i])*np.log(1-self.sigmoide(np.dot(np.transpose(self.W), np.reshape(X[i],(3073,1))))) for i in range(num_train)]
        ThirdTerm = [self.W[j]**2 for j in range(num_features)]
        cost = -np.sum(FirstTerm)-np.sum(SecondTerm) + (reg/2)*np.sum(ThirdTerm)
        loss = cost * (1/num_train)  # On divide par le nombre de samples

        sigmaVector = np.transpose([self.sigmoide(np.dot(np.transpose(self.W), np.reshape(x,(3073,1)))) for x in X])
        sigmaVector = sigmaVector.reshape(num_train,1)
        y = y.reshape(num_train,1)
        dW = np.dot(np.transpose(X),(sigmaVector-y)) + reg*self.W
        dW = dW/num_train
        diagSigma = np.zeros((num_train,num_train))

        for i in range(num_train):
            diagSigma[i][i] = self.sigmoideDerivative(np.dot(np.transpose(self.W), np.reshape(X[i],(3073,1))))  #I think it cost less computation if i do a for loops in range of num-train than a list comprenhension in range num_train x num_features

        ddW = np.dot(np.dot(np.transpose(X),diagSigma),X)
        ddW = ddW/num_train
        return loss, dW, ddW

    def predict(self, X):
        num_train = X.shape[0]
        sigmaVector = np.transpose([self.sigmoide(np.dot(np.transpose(self.W), np.reshape(X[i], (3073, 1)))) for i in range(num_train)])
        sigmaVector = sigmaVector.reshape(num_train, 1)
        y_pred = [1 if sigmaVector[i]>0.5 else 0 for i in range(num_train)]
        return y_pred

