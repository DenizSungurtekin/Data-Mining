import numpy as np

class Classifier(object):
    def __init__(self, classifier_type):
        self.classifier_type = classifier_type
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
                batch_size=200, verbose=False, newton = False):

        num_train, dim = X.shape
        num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            if self.classifier_type == 'logistic':
                self.W = 0.001 * np.random.randn(dim, 1)
            else:
                self.W = 0.001 * np.random.randn(dim, num_classes)
        # Run stochastic gradient descent to optimize W
        loss_history = []
        acc_history = []
        for it in range(1, num_iters+1):
            X_batch = None
            y_batch = None

            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]

            # evaluate loss and gradient
            loss, grad,hessian = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update using SGD
            if newton == False:

                self.W = self.W - learning_rate*grad
            else:
                h = np.matmul(np.linalg.pinv(hessian),grad)
                self.W = self.W - learning_rate*h

            acc_history.append(np.mean(self.predict(X_batch) == y_batch))

            if verbose and it%100 == 0:
                print('iteration {} / {} : loss {}'.format(it, num_iters, loss), end='\r')
        if verbose:
            print(''.ljust(70), end='\r')

        return loss_history, acc_history




 