
import random
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_IRIS, load_CIFAR10
import matplotlib.pyplot as plt
from logistic_regression import Logistic


class LogisticTEST:

    def __init__(self):
        self.setup_data()

    def setup_data(self):
        # Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
        try:
           del self.X_train, self.y_train
           del self.X_test, self.y_test
           print('Clear previously loaded data.')
        except:
           pass

        # load 2 classes
        cifar10_dir = 'datasets/cifar-10-batches-py'
        classes=['horse', 'car']
        self.X_train, self.y_train, self.X_test, self.y_test = load_CIFAR10(cifar10_dir, classes=['horse', 'car'])

        # Visualize some examples from the dataset.
        # We show a few examples of training images from each class.
        print("Visualizing some samples")
        num_classes = len(classes)
        samples_per_class = 7
        for y, cls in enumerate(classes):
            idxs = np.flatnonzero(self.y_train == y)
            idxs = np.random.choice(idxs, samples_per_class, replace=False)
            for i, idx in enumerate(idxs):
                plt_idx = i * num_classes + y + 1
                plt.subplot(samples_per_class, num_classes, plt_idx)
                plt.imshow(self.X_train[idx].astype('uint8'))
                plt.axis('off')
                if i == 0:
                    plt.title(cls)
        plt.show()

        # choising parameters for subsampling
        num_training = 9000
        num_validation = 1000

        # subsample the data
        mask = list(range(num_training, num_training + num_validation))
        X_val = self.X_train[mask]
        y_val = self.y_train[mask]
        mask = list(range(num_training))
        self.X_train = self.X_train[mask]
        self.y_train = self.y_train[mask]

        # Preprocessing: reshape the image data into rows
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], -1))
        X_val = np.reshape(X_val, (X_val.shape[0], -1))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], -1))

        # Normalize the data: subtract the mean image and divide by the std
        mean_image = np.mean(self.X_train, axis = 0)
        std_image = np.std(self.X_train, axis = 0)
        self.X_train -= mean_image
        self.X_train /= std_image
        X_val -= mean_image
        X_val /= std_image
        self.X_test -= mean_image
        self.X_test /= std_image

        # add bias dimension and transform into columns
        self.X_train = np.hstack([self.X_train, np.ones((self.X_train.shape[0], 1))])
        X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
        self.X_test = np.hstack([self.X_test, np.ones((self.X_test.shape[0], 1))])

        num_dim = self.X_train.shape[1]

        # Printing dimensions
        print('Train data shape: ', self.X_train.shape)
        print('Train labels shape: ', self.y_train.shape)
        print('Validation data shape: ', X_val.shape)
        print('Validation labels shape: ', y_val.shape)
        print('Test data shape: ', self.X_test.shape)
        print('Test labels shape: ', self.y_test.shape)

    def test_scores(self):
        dim = self.X_train.shape[1]
        logistic_regression = Logistic(random_seed=123)
        logistic_regression.W = 0.001 * np.random.randn(dim, 1)
        scores = logistic_regression.loss(self.X_train)
        if scores is None:
            print("You have to implement scores first.")
        else:
            if  np.abs(np.sum(scores) - 4497.79763431513) < 1e-5:
                print("Great! Your implementation of scores seems good !")
            else:
                print("Bad news! Your implementation of scores seems wrong !")

    def test_loss(self):
        dim = self.X_train.shape[1]
        logistic_regression = Logistic(random_seed=123)
        logistic_regression.W = 0.001 * np.random.randn(dim, 1)
        loss, _ , _= logistic_regression.loss(self.X_train, self.y_train, 0.0)
        print(np.abs(loss))
        if loss is None:
            print("You have to implement loss first.")
        else:
            if np.abs(loss - 0.6933767017840157) < 1e-2:
                print("Great! Your implementation of the loss seems good !")
            else:
                print("Bad news! Your implementation of  the loss seems wrong !")

    def test_gradient(self):
        dim = self.X_train.shape[1]
        logistic_regression = Logistic(random_seed=123)
        logistic_regression.W = 0.001 * np.random.randn(dim, 1)
        _, grad, _ = logistic_regression.loss(self.X_train, self.y_train, 0.0)
        if not np.sum(grad):
            print("You have to implement the gradients first.")
        else:
            if np.abs(np.sum(grad) - 26.725907504626434) < 1e-5:
                print("Great! Your implementation of gradients seems good !")
            else:
                print("Bad news! Your implementation of gradients seems wrong !")


    def test_hessian(self):
        dim = self.X_train.shape[1]
        logistic_regression = Logistic(random_seed=123)
        logistic_regression.W = 0.001 * np.random.randn(dim, 1)
        _, _, hessian = logistic_regression.loss(self.X_train, self.y_train, 0.0)

        print(hessian)
        print(hessian.shape)

    def test_classifier(self):
        dim = self.X_train.shape[1]
        logistic_regression = Logistic(random_seed=123)
        logistic_regression.W = 0.001 * np.random.randn(dim, 1)
        y_pred = logistic_regression.predict(self.X_train)
        print(np.sum(y_pred))
        print("error:",abs((np.sum(y_pred) - 4718.0)))
        if not np.sum(y_pred):
            print("You have to implement prediction first.")
        else:
            if np.abs(np.sum(y_pred) - 4718.0) < 1e-7:
                print("Great! Your implementation of prediction seems good !")
            else:
                print("Bad news! Your implementation of prediction seems wrong !")

    def tune_hyperparameters(self):
        from logistic_regression import Logistic
        import copy


        self.best_hist_gd = []
        self.best_logistic_gd = None
        best_val_gd = -1
        self.best_hist_nw = []
        self.best_logistic_nw = None
        best_val_nw = -1

        learning_rates = [1e-8, 1e-6, 1e-2]
        regularization_strengths = [1e-4, 1e-2, 0, 1e2, 1e4]

        verbose = True
        newton_iteration = [[False,300],[True,10]] # [Gradient iteriation,Newton iteration] HERE YOU CAN CHANGE THE NUMBER OF ITERATION
        for nw in newton_iteration:
            for lr in learning_rates:
                for reg in regularization_strengths:
                    print("lr = {}, reg = {}, newton = {} ".format(lr, reg, nw))
                    model = Logistic(random_seed=123)
                    model.train(self.X_train,self.y_train,lr,reg,nw[1],newton=nw[0])

                    acc_train = np.mean(model.predict(self.X_train)==self.y_train)
                    acc_val = np.mean(model.predict(self.X_test)==self.y_test)
                    if nw[0]:
                        if acc_val > best_val_nw:
                            best_val_nw = acc_val
                            self.best_hist_nw.append(best_val_nw)
                            self.best_logistic_nw = copy.deepcopy(model)
                    else:
                        if acc_val > best_val_gd:
                            best_val_gd = acc_val
                            self.best_hist_gd.append(best_val_gd)
                            self.best_logistic_gd = copy.deepcopy(model)

                    print("\r\t -> train acc = {:.3f}, val acc = {:.3f}".format(acc_train, acc_val))


        print('best validation GD: {:.3f}'.format(best_val_gd))
        print('best validation NW: {:.3f}'.format(best_val_nw))


        print("done ", end='\r')



    # def compute_accuracy(self):                                       # Not used -> Done before
    #     # evaluate bast model using Gradient Descent on test set
    #     self.y_test_pred = self.best_logistic_gd.predict(self.X_test)
    #     test_accuracy = 0 # to replace
    #     print('Logistic on raw pixels final test set accuracy using Gradient Descent: {:.3f}'.format(test_accuracy))

    # def evaluate(self):                                               # Not used -> Done before
    #     # evaluate bast model using Newton's Method on test set
    #     self.y_test_pred = self.best_logistic_nw.predict(self.X_test)
    #     test_accuracy = 0 # to replace
    #     print('Logistic on raw pixels final test set accuracy using Newtons Method: {:.3f}'.format(test_accuracy))


    def plot_losses(self):
        plt.plot(self.best_hist_gd,label="Gradient Descent")
        plt.plot(self.best_hist_nw,label="Newton")
        plt.legend()
        plt.show()
        # ## Plot the history of losses for the best models (best_hist_gd, best_hist_nw)

if __name__ == '__main__':      #IMPORTANT YOU CAN COMMENT OR DECOMMENT THE FOLLOWING FUNCTION TO RUN IT SEPARATELY  THE LAST TWO ARE NOT USED
    t = LogisticTEST()

    t.test_scores()
    t.test_loss()

    t.test_gradient()
    t.test_hessian()
    t.test_classifier()
    t.tune_hyperparameters()
    t.plot_losses()
    # t.compute_accuracy()
    # t.evaluate()
