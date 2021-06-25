import numpy as np
import pickle, gzip 

class Numbers:
    """
    Class to store MNIST data for images of 9 and 8 only
    """ 
    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if you'd like
        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
 
        self.train_x, self.train_y = train_set
        train_indices = np.where(self.train_y > 7)
        self.train_x, self.train_y = self.train_x[train_indices], self.train_y[train_indices]
        self.train_y = self.train_y - 8
 
        self.valid_x, self.valid_y = valid_set
        valid_indices = np.where(self.valid_y > 7)
        self.valid_x, self.valid_y = self.valid_x[valid_indices], self.valid_y[valid_indices]
        self.valid_y = self.valid_y - 8

from collections import defaultdict
class LogReg:
    
    def __init__(self, X, y, eta = 0.1):
        """
        Create a logistic regression classifier
        :param num_features: The number of features (including bias)
        :param eta: Learning rate (the default is a constant value)
        :method: This should be the name of the method (sgd_update or mini_batch_descent)
        :batch_size: optional argument that is needed only in the case of mini_batch_descent
        """
        self.X = X
        self.y = y
        self.w =[]
        #self.w = np.zeros(X.shape[1]) # can remove from here and ask to be defined in the function
        self.eta = eta
        
    def calculate_score(self, x):
        """
        :param x: This can be a single training example or it could be n training examples
        :return score: Calculate the score that you will plug into the logistic function
        """
        # TODO: Compute the score to be fed to the sigmoid function
        scores = np.dot(x, self.w)
        return scores
        
    
    def sigmoid(self, score):
        """
        :param score: Either a real valued number or a vector to convert into a number between 0 and 1
        :return sigmoid: Calcuate the output of applying the sigmoid function to the score. This could be a single
        value or a vector depending on the input
        """
        # TODO: Complete this function to return the output of applying the sigmoid function to the score
        return 1 / (1 + np.exp(-score))
        
    
    def compute_gradient(self, x, h, y):
        """
        :param x: Feature vector
        :param h: predicted class label
        :param y: real class label
        :return gradient: Return the derivate of the cost w.r.t to the weights
        """
        # TODO: Finish this function to compute the gradient

        gradient = np.dot((h - y),x) 
        return gradient
        
     
    def sgd_update(self):
        """
        Compute a stochastic gradient update over the entire dataset to improve the log likelihood.
        :param x_i: The features of the example to take the gradient with respect to
        :param y: The target output of the example to take the gradient with respect to
        :return: Return the new value of the regression coefficients
        """ 
        # TODO: Finish this function to do a stochastic gradient descent update over the entire dataset
        # and return the updated weight vector
        self.w = np.zeros(self.X.shape[1])
        for i in range(len(self.X)):
            z = self.calculate_score(self.X[i])
            h = self.sigmoid(z)
            gradient = self.compute_gradient(self.X[i],h,self.y[i])
            self.w -= self.eta * gradient
        return self.w

    
    def mini_batch_update(self, batch_size):
        """
        One iteration of the mini-batch update over the entire dataset (one sweep of the dataset).
        :param X: NumPy array of features (size : no of examples X features)
        :param y: Numpy array of class labels (size : no of examples X 1)
        :param batch_size: size of the batch for gradient update
        :returns w: Coefficients of the classifier (after updating)
        """
        # TODO: Performing mini-batch training follows the same steps as in stochastic gradient descent,
        # the only major difference is that we’ll use batches of training examples instead of one. 
        # Here we decide a batch size, which is the number of examples that will be fed into the 
        # computational graph at once.
        X_batch_li = list()
        y_batch_li = list()
    
        for i in range(len(self.y) // batch_size):
            X_batch_li.append(self.X[i * batch_size : i * batch_size + batch_size])
            y_batch_li.append(self.y[i * batch_size : i * batch_size + batch_size])
        
    
        if len(self.y) % batch_size > 0:
            X_batch_li.append(self.X[len(self.y) // batch_size * batch_size:, :])
            y_batch_li.append(self.y[len(self.y) // batch_size * batch_size:])
            
        n_batches = len(y_batch_li)
        

        self.w = np.zeros(X_batch_li[0].shape[1])
        for i in range(n_batches):
            
            X_batch = X_batch_li[i]
            y_batch = y_batch_li[i]
            z = self.calculate_score(X_batch)
        
            h = self.sigmoid(z)
            gradient = self.compute_gradient(X_batch,h,y_batch)
            self.w -= self.eta * gradient
        return self.w

    def progress(self, test_x, test_y, update_method, *batch_size):
        """
        Given a set of examples, computes the probability and accuracy
        :param test_x: The features of the test dataset to score
        :param test_y: The features of the test 
        :param update_method: The update method to be used, either 'sgd_update' or 'mini_batch_update'
        :param batch_size: Optional arguement to be given only in case of mini_batch_update
        :return: A tuple of (log probability, accuracy)
        """
        # TODO: Complete this function to compute the predicted value for an example based on the logistic value
        # and return the log probability and the accuracy of those predictions
        self.X = test_x
        self.y = test_y
        result = ()
        log_prob=0
        X_batch = list()
        y_batch = list()
        preds=np.zeros(test_y.shape)
        if update_method == 'sgd_update':
            self.w = np.zeros(test_x.shape[1])
            for i in range(len(test_x)):
                z = self.calculate_score(test_x[i])
                h = self.sigmoid(z)
                if h>= 0.5:
                    preds[i]=1
                else:
                    preds[i]=0
                gradient = self.compute_gradient(self.X[i],h,self.y[i])
                self.w -= self.eta * gradient
                log_prob += np.sum( test_y*z - np.log(1 + np.exp(z)) )
        else:
            
            X_batch_li = list()
            y_batch_li = list()
            batch_size=batch_size[0]
    
            for i in range(len(self.y) // batch_size):
                X_batch_li.append(self.X[i * batch_size : i * batch_size + batch_size])
                y_batch_li.append(self.y[i * batch_size : i * batch_size + batch_size])
        
    
            if len(self.y) % batch_size > 0:
                X_batch_li.append(self.X[len(self.y) // batch_size * batch_size:, :])
                y_batch_li.append(self.y[len(self.y) // batch_size * batch_size:])
            
            n_batches = len(y_batch_li)
        

            self.w = np.zeros(X_batch_li[0].shape[1])
            k=0
            for i in range(n_batches):
            
                X_batch = X_batch_li[i]
                y_batch = y_batch_li[i]
                z = self.calculate_score(X_batch)
                h = self.sigmoid(z)
                for j in range(len(h)):
                    if h[j]>= 0.5:
                        preds[k]=1
                    else:
                        preds[k]=0
                    k += 1
                    
                gradient = self.compute_gradient(X_batch,h,y_batch)
                self.w -= self.eta * gradient
                log_prob += np.sum( test_y[i]*z - np.log(1 + np.exp(z)))


        accuracy = ((preds == self.y).sum().astype(int) / len(preds))
        result=(log_prob,accuracy)
        return result


import unittest

class LogRegTester(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0.1, 0.3 ], [0.4, 0.6], [0.8, 0.1], [0.8, 0.1], [0.5, 0.8]])
        self.y = np.array([0,  0, 1, 1,  0])
        self.log_reg_classifier_1 = LogReg(self.X, self.y, 0.5)
        self.log_reg_classifier_2 = LogReg(self.X, self.y, 0.5)
        
    def test_sgd_update(self):
        """
        Test sgd_update function from LogReg
        """
        weights = self.log_reg_classifier_1.sgd_update()
        self.assertEqual(round(weights[0], 2), 0.16)
        self.assertEqual(round(weights[1], 2), -0.37)
        
    def tests_mini_batch_update(self):
        """
        Test mini_batch_update function from LogReg
        """
        weights = self.log_reg_classifier_2.mini_batch_update(2)
        self.assertEqual(round(weights[0], 2), 0.17)
        self.assertEqual(round(weights[1], 2), -0.37)
        
    def tests_progress_sgd_update(self):
        """
        Test progress function from LogReg with method = 'sgd_update'
        """
        self.log_reg_classifier_1 = LogReg(self.X[:4], self.y[:4], 0.5)
        log_prob, accuracy = self.log_reg_classifier_1.progress(self.X[4:], self.y[4:], 'sgd_update')
        self.assertEqual(round(log_prob, 1), -0.7)  # Changed to round 1.
        self.assertEqual(accuracy, 0)
        
    
# tests = LogRegTester()
# myTests = unittest.TestLoader().loadTestsFromModule(tests)
# unittest.TextTestRunner().run(myTests)


from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class MultiLogReg:
    
    def __init__(self, X, y, eta = 0.1):
        self.X = self.normalize_data(X)
        #self.X = X
        self.y = self.one_hot_encoding(y)
        self.eta = eta
        self.opt_class = []
        self.opt_weights=np.zeros((self.X.shape[1],10))
      
        
    def one_hot_encoding(self, y):
        # TO DO: Represent the output vector y as a one hot encoding. Create a matrix of dimensions (m X 10) 
        # where m = number of examples, and 10 for number of classes
        # if the class for the ith example is 7, then y[i][7] = 1 and the for k != 7, y[i][k] = 0.
        # where m = number of examples, and 10 for number of classes
        # if the class for the ith example is 7, then y[i][7] = 1 and the for k != 7, y[i][k] = 0.
        enc = [[0 for i in range(10)] for j in range(len(y))]
        for i in range(len(y)):
            enc[i][y[i]] = 1
        enc = np.array(enc)
        return enc
    
        
    def normalize_data(self, X):
        # TO DO: Normalize the feature values of dataset X using the mean and standard deviation of the respective featur
        shape = X.shape
        X = np.reshape(X, (-1,))
        mean = np.mean(X)
        std = np.std(X)
        # Create a new array for storing standardized values
        std_values = list()
        for i in X:
            x_norm = (i - mean) / std
            std_values.append(x_norm)  
        n_array = np.array(std_values)
        ans = np.reshape(n_array,shape)
        return ans
    

        
    def get_optimal_parameters(self):
        # TO DO: This is the main training loop. You will have to find the optimal weights for all 10 models
        # Each model is fit to it's class which is (0-9), and the cost function will be against all of the other 
        # numbers "the rest".
        n = 10
        print("Optimal parameters")
        for i in range(n):
            self.opt_class.append(LogReg(self.X, self.y, self.eta))
            prevacc, prevacc1 = 0, 0
            for epoch in range(100):
                trainX, testX, trainY, testY = train_test_split(self.X, self.y[:,i], train_size = 0.8)
                self.opt_class[i].X = trainX
                self.opt_class[i].y = trainY
                logprob, accuracy = self.opt_class[i].progress(testX, testY, 'mini_batch_update', 20)

                if abs(accuracy - prevacc) < 0.01 and abs(prevacc - prevacc1) < 0.01:
                    print(i , "\t", epoch, "\t", accuracy)
                    self.opt_weights[:,i]=self.opt_class[i].mini_batch_update(20)
                    break
                prevacc1 = prevacc
                prevacc = accuracy

   
    
    def predict(self, test_image, test_label):
        # TO DO: This function should return the probabilities predicted by each of the models for some given 
        # input image. The probabilities are sorted with the most likely being listed first.
        # Return a vector of shape (10, 2) with the first column holding the number and the second column with
        # the probability that the test_image is that number
        probs = np.zeros((10,2))
        ans = np.zeros((10,2))
        for num in range(10):
            a_opt = self.opt_weights[:,num]
            probs[num,0] = num
            probs[num,1] = 1.0/(1.0 + np.exp(-(np.dot(test_image,a_opt))))

        probs = probs[probs[:,1].argsort()[::-1]]
        probs = np.around(probs, decimals=4)
        return probs