from __future__ import print_function

import numpy as np
import copy as cp


class NN(object):
    """
    A two-layer fully-connected neural network. The network has an input dimension
    of N, a hidden layer dimension of H, and performs classification over C
    classes.  We train the network with a softmax loss function.
    The network uses a sigmoid activation function
    after the first fully connected layer.
    In other words, the network has the following architecture:
    input - fully connected layer - sigmoid - fully connected layer - softmax
    The outputs of the second fully-connected layer are the scores for each
    class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)
        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def sigmoid(self, a):
        return np.divide(1,(1+np.exp(-a)))

    def sigmoid_derivative(self, a):
        return np.multiply(self.sigmoid(a),(1-self.sigmoid(a)))

    def stable_softmax(self, a):
        """Compute the softmax of vector x in a numerically stable way."""
        shiftx = a - np.max(a)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)

    def loss(self, X, y=None):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.
        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each
            y[i] is an integer in the range 0 <= y[i] < C. This parameter is
            optional; if it is not passed then we only return scores, and if it
            is passed then we instead return the loss and gradients.
        Returns:
        If y is None, return a matrix scores of shape (N, C) where
        scores[i, c] is the score for class c on input X[i].
        If y is not None, instead return a tuple of:
        - loss: Loss for this batch of training samples.
        - grads: Dictionary mapping parameter names to gradients of those
            parameters with respect to the loss function; has the same keys as
            self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        
        ###########################
        # TODO: Perform the forward pass,
        # computing the class scores for the input.
        # Aplly sigmoid activation function at the hidden layer
        # Store the result in the scores variable,
        # which should be an array of shape (N, C).
        ##########################################

        Z1 = np.dot(X,W1) + b1          # dot product of X (input) and first set of  weights
        a1 = self.sigmoid(Z1)           # apply sigmoid activation function
        Z2 = np.dot(a1,W2) + b2         # dot product of hidden layer (a1) and second set of weights
        
        ##########################
        #         END OF YOUR CODE
        ##########################

        # If the targets are not given then jump out, we're done
        if y is None:
            return Z2

        # scaling (in terms of exponential) for numerical stability
        scores = Z2 - np.max(Z2, axis=1).reshape((-1, 1))

        # compute the loss.
        # We store the result in the variable loss, which should be a scalar. 
        # We use the Softmax classifier loss.
        
        loss = None
        exp_scores = np.exp(scores)
        probabilitites = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True) # Softmax
        loss = -np.sum(np.log(probabilitites[range(N), y])) #data loss
        loss /= N


        # Backward pass: compute gradients
        grads = {}

        ############################
        # TODO: Compute the backward pass,
        # computing the derivatives of the weights
        # and biases. Store the results in the grads dictionary.
        # For example,
        # grads['W1'] should store the gradient on W1,
        # and be a matrix of same size
        #############################

        grad_Z2 = cp.deepcopy(probabilitites) # Softmax    # N x C
        grad_Z2[range(N), y] -= 1
        grad_Z2 /= N

        grad_Z1 = np.multiply(self.sigmoid_derivative(Z1), np.matmul(Z2, W2.T))

        grads['b1'] = np.sum(grad_Z1, axis=0) / N  # adjusting first set (input --> hidden) bias
        grads['W1'] = np.dot(X.T, grad_Z1)  # adjusting first set (input --> hidden) weights

        grads['b2'] = np.sum(grad_Z2, axis=0) / N  # adjusting second set (hidden --> output) bias
        grads['W2'] = np.dot(a1.T, grad_Z2)  # adjusting second set (hidden --> output) weights

        ###########################
        # END OF YOUR CODE
        ##########################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1, learning_rate_decay=0.95, num_iters=500,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means
            that X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving val labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning
            rate after each epoch.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters 
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None


            batch_ind = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_ind, :]
            y_batch = y[batch_ind]


            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch)
            # STOPPING CRITERION
            if len(loss_history) != 0:
                if abs(loss - loss_history[-1]) < 0.0001:
                    break
            loss_history.append(loss)

            ##################################################################
            # TODO: Use the gradients in the grads dictionary to update the
            # parameters of the network (stored in the dictionary self.params)
            # using stochastic gradient descent. You'll need to use the
            # gradients stored in the grads dictionary defined above.
            ##################################################################
            self.params['W1'] -= learning_rate*grads['W1'] #- np.multiply(grads['W1'],self.params['W1'])
            self.params['b1'] -= learning_rate*grads['b1'] #- np.multiply(grads['b1'],self.params['b1'])
            self.params['W2'] -= learning_rate*grads['W2'] #- np.multiply(grads['W2'],self.params['W2'])
            self.params['b2'] -= learning_rate*grads['b2'] #- np.multiply(grads['b2'],self.params['b2'])

            #################################################################
            #                             END OF YOUR CODE
            #################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and validation accuracy and decay learning rate
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels
        for data points. For each data point we predict scores for each of the
        C classes, and assign each data point to the class with the highest
        score.
        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points
            to classify.
        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each
            of the elements of X. For all i, y_pred[i] = c means that X[i] is
            predicted to have class c, where 0 <= c < C.
        """
        y_pred = None

        #############################################################
        # TODO: Implement this function;   #
        #############################################################
        # Note on scores array
        # If y is None, return a matrix scores of shape (N, C) where
        # scores[i, c] is the score for class c on input X[i
        scores = self.loss(X)
        exp_scores = np.exp(scores)
        probabilitites = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        y_pred = np.argmax(probabilitites, axis=1)


        
        #############################################################
        #                              END OF YOUR CODE             #
        #############################################################

        return y_pred