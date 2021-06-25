# *** basic k-means clustering algorithms (w/Elbow analysis) and helper functions ***
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import unittest

class DataIris:
    def __init__(self):
        data = load_iris()
        self.data = data.data[:, [1, 3]]
        self.target = data.target


class KMeansClustering:
    def __init__(self, X, K):
        '''
        Params : 
            X : (np.ndarray) of dimension (N, d) N is the number of points
            K : (int) number of means/centroids to evaluate
            epochs : (int) maximum number of epochs to evaluate for the centroids
        '''
        self.X = X
        self.K = K
        self.centroids = self.initialize_centroids()
        
        
    def initialize_centroids(self):
        '''
        Randomly select K distinct points from the dataset in self.X
        Params : 
            None
        RETURN :
            means : (np.ndarray) of the dimension (K, d)
        '''
        self.centroids = self.X.copy()
        np.random.shuffle(self.centroids)
        return self.centroids[:self.K]
    
    def compute_distances(self):
        '''
        Comupute a distance matrix of size (N, K) where each cell (i, j) represents the distance between 
        i-th point and j-th centroid. We shall use Euclidean distance here.
        
        PARAMS:
            centroids : (np.ndarray) of the dimension (K, d)
        RETURN:
            distance_matrix : (np.ndarray) of the dimension (N, K)
        '''
        distance_matrix = np.sqrt(((self.X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return distance_matrix
    
    def compute_cluster_assignment(self, distance_matrix):
        '''
        Comupute a distance matrix of size (N, K) where each cell (i, j) represents the distance between 
        i-th point and j-th centroid. We shall use Euclidean distance here.
        
        PARAMS:
            distance_matrix : (np.ndarray) of the dimension (N, K)
        RETURN:
            labels : (np.ndarray) of the size (N)
        '''
        labels = np.argmin(distance_matrix, axis=0)
        return labels
    
    def compute_centroids(self, labels):
        '''
        Randomly select K distinct points from the dataset in self.X
        Params : 
            labels : (np.ndarray) of the dimension (N) where each i-th item reperesents the closest
            centroid among the K centroids. Each value here must be between 0 and K-1.
        RETURN :
            updated_means : (np.ndarray) of the dimension (K, d)
        '''
        updated_means = np.array([self.X[labels==k].mean(axis=0) for k in range(self.centroids.shape[0])])
        return updated_means
    
    def convert_to_2d_array(self,points):
        """
        Converts `points` to a 2-D numpy array.
        """
        points = np.array(points)
        if len(points.shape) == 1:
            points = np.expand_dims(points, -1)
        return points
    
    def SSE(self,points):
        """
        Calculates the sum of squared errors for the given list of data points.
        """
        points = self.convert_to_2d_array(points)
        centroid = np.mean(points, 0)
        errors = np.linalg.norm(points-centroid, ord=2, axis=1)
        return np.sum(errors)
    
    def distance(self, a, b, ax=1):
        return np.linalg.norm(a - b, axis=ax)
   
    
    def cluster(self, epochs):
        '''
        Implement the K-means algorithm here as described above. However loop for a maximum of self.epochs.
        Ensure that you have a condition that checks whether the epochs have changed since the last epoch or not
        For this use a threshold of change of 0.01.
        
        PARAMS:
            epochs : (integer) maximum number of epochs
        RETURN:
            centroids : (np.ndarray) of the size (K, d) also store in a class variable self.centroids
        '''
        best_sse = np.inf
        for ep in range(epochs):
            distance_matrix = np.sqrt(((self.X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distance_matrix, axis=0)
            new_centroids = np.array([self.X[labels==k].mean(axis=0) for k in range(self.centroids.shape[0])])
            err = self.distance(self.centroids, new_centroids, None)
            if err <= 0.01:
                break
            self.centroids = new_centroids
            self.show_progress(ep)
        return self.centroids, labels
            
            
    
    def show_progress(self, epoch):
        '''
        PARAMS:
            epoch : (integer) tells which epoch is it
        RETURN:
            None
        '''
        plt.plot(self.X[:, 0], self.X[:, 1], 'o', color='y')
        for i in range(self.K):
            plt.plot(self.centroids[i, 0], self.centroids[i, 1], 'o', color='k')
        plt.title('Centroids at epoch : {}'.format(epoch))
        plt.show()
        for i in range(self.K):
            plt.plot(self.centroids[i, 0], self.centroids[i, 1], 'o', color='k')
        plt.title('Centroids at epoch : {}'.format(epoch))
        plt.show()

    def elbow_analysis(self, k_range):
        '''
        PARAMS:
            k_range : (list of +ve integers) contains the K number of hyperparameters k to peform 
            the analysis over
        RETURN:
            avg_variance : (list of float) list od size K. contains the average variance of clusters corresponding to each 
            to each hyperparameter k
        '''
        C, avg_variance = [0 for k in range(k_range)], [0 for k in range(k_range)]
        for k in range(1, k_range):
            for i in range(20):
                self.K = k
                self.initialize_centroids()
                curr_centroids = self.centroids
                newCentroids = np.zeros(self.K)
                distance_matrix = self.compute_distances()
                labels = self.compute_cluster_assignment(distance_matrix)
                newCentroids = self.compute_centroids(labels)
                self.centroids = newCentroids

            var = 0
            
            for j in range(len(self.X)):
                points = []
                if labels[j] == i:
                    points.append(self.X[j])
                points = np.array(points)
                for m in range(len(points)):
                    dist = 0
                    dist = self.distance(points[m], newCentroids[i], None)
                    var += (dist ** 2)
                var = var / k
            avg_variance[k]=(var)
        return avg_variance


class KMeansTester(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0.1, 0.3], [0.4, 0.6], [0.2, 0.4], [3.1, 3.1], [3.5, 2.9]])
        self.cluster_obj_1 = KMeansClustering(self.X, 2)
        self.cluster_obj_2 = KMeansClustering(self.X, 2)
        self.cluster_obj_3 = KMeansClustering(self.X, 2)
        self.cluster_obj_4 = KMeansClustering(self.X, 2)
        
    def test_initialize_centroids(self):
        """
        Test initialize_centroids function from KMeansClustering
        """
        means = self.cluster_obj_1.initialize_centroids()
        self.assertEqual(means.shape[0], 2)
        self.assertEqual(means.shape[1], 2)
        
    def test_compute_distances(self):
        """
        Test compute_distances function from KMeansClustering
        """
        self.cluster_obj_2.centroids = self.X[:2, :]
        distance_matrix = self.cluster_obj_2.compute_distances()
        self.assertEqual(round(distance_matrix[0,0], 2), 0.0)
        self.assertEqual(round(distance_matrix[0,1], 2), 0.42)
        
    def tests_compute_cluster_assignment(self):
        """
        Test compute_cluster_assignment function from KMeansClustering
        """
        self.cluster_obj_3.centroids = self.X[:2, :]
        distance_matrix = self.cluster_obj_3.compute_distances()
        labels = self.cluster_obj_3.compute_cluster_assignment(distance_matrix)
        self.assertEqual(labels[0], 0)
        self.assertEqual(labels[1], 1)
        self.assertEqual(labels[2], 0)
        
    def test_compute_centroids(self):
        """
        Test compute_centroids function from KMeansClustering
        """
        self.cluster_obj_4.centroids = self.X[:2, :]
        distance_matrix = self.cluster_obj_4.compute_distances()
        labels = self.cluster_obj_4.compute_cluster_assignment(distance_matrix)
        new_means = self.cluster_obj_4.compute_centroids(labels)
        self.assertEqual(round(new_means[0, 0], 2), 0.15)
        self.assertEqual(round(new_means[0, 1], 2), 0.35)

# tests = KMeansTester()
# myTests = unittest.TestLoader().loadTestsFromModule(tests)
# unittest.TextTestRunner().run(myTests)


class GaussianMixtureModel1D:
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.mean, self.variance, self.weight = self.initialize_parameters()
        
    def initialize_parameters(self):
        mean = np.random.choice(self.X, self.K)
        variance = np.random.random_sample(size=self.K) * 2
        weights = np.ones(self.K) / self.K
        return mean, variance, weights
    
    def compute_pdf(self, x, k):
        '''
        Evaluate the p.d.f value for 1-D point i.e scalar value for the w.r.t to the k-th cluster
        Params : 
            x : (float) the point
            k : (integer) the k-th elements from mean, variance and weights correspond to k-th cluster parameters.
                Use those to estimate your result.
        RETURN :
            result : (float) evalutated using the formula described above
        '''
        a = 1/(np.sqrt(2*np.pi*self.variance[k]))
        b = np.exp(-(np.square(x - self.mean[k])/(2*self.variance[k])))
        return a * b
        
    def compute_pdf_matrix(self):
        '''
        Evaluate the p.d.f martix by calling compute_pdf() for each combination of x and k
        Params : 
            None
        RETURN :
            result : (np.array) matrix of size N X K containing p.d.f values
        '''
        pdf_matrix = []
        for j in range(self.K):
            pdf_matrix.append(self.compute_pdf(self.X,j))
        return ((np.array(pdf_matrix)).transpose())
    
    def compute_posterior(self, pdf_matrix):
        '''
        Evaluate the posterior probability martix as described by the formula above
        Params : 
            pdf_matrix : (np.array) matrix of size N X K containing p.d.f values
        RETURN :
            result : (np.array) matrix of size N X K containing posterior probability values
        '''
        b = []
        pdf_matrix=pdf_matrix.transpose()
        for j in range(self.K):
            b.append((pdf_matrix[j] * self.weight[j]) / (np.sum([pdf_matrix[i] * self.weight[i] for i in range(self.K)], axis=0)))
        return (np.array(b).transpose())
    
    def reestimate_params(self, posterior_matrix):
        '''
        Re-estimate the cluster parameters as described by the formulae above and 
        store them in their respective class variables
        Params : 
            posterior_matrix : (np.array) matrix of size N X K containing posterior probability values
        RETURN :
            None
        '''
        mean=[]
        variance=[]
        weight=[]
        posterior_matrix=posterior_matrix.transpose()
        for j in range(self.K):
            mean.append(np.sum(posterior_matrix[j] * self.X) / (np.sum(posterior_matrix[j])))
            variance.append (np.sum(posterior_matrix[j] * np.square(self.X - mean[j])) / (np.sum(posterior_matrix[j])))
            weight.append(np.mean(posterior_matrix[j]))
          
        self.mean = np.array(mean)
        self.variance =  np.array(variance)
        self.weight = np.array(weight)

            
    def exp_maximize(self, epochs):
        '''
        Peform the expectation-maximization method as dicussed above by calling the functions in their 
        respective sequence. Also plot the progress of the process by calling the plot_progress function
        after every regular interval of epochs.
        Params : 
            epochs : (integer) maximum number of epochs to run the loop for
        RETURN :
            None
        '''
        for i in range(epochs):
            pdf_matrix = self.compute_pdf_matrix()
            posterior_matrix = self.compute_posterior(pdf_matrix)
            self.reestimate_params(posterior_matrix)
            if i % 1 == 0:
                self.plot_progress()
        
        
    
    def plot_progress(self):
        points = np.linspace(np.min(self.X),np.max(self.X),500)
        plt.figure(figsize=(10,4))
        plt.xlabel("$x$")
        plt.ylabel("pdf")
        plt.plot(self.X, 0.1*np.ones_like(self.X), 'x', color='navy')
        for k in range(self.K):
            plt.plot(points, [self.compute_pdf(p, k) for p in points])
        plt.show()

import unittest

class GMMTester(unittest.TestCase):
    def setUp(self):
        self.X = np.array([0.1, 1.2, 0.3, 0.4, 0.3, 3.5, 2.9])
        self.means = [-2.0, 2.5]
        self.variances = [1.0, 1.3]
        self.weights = [0.1, 0.2]
        
        self.cluster_obj_1 = GaussianMixtureModel1D(self.X, 2)
        self.cluster_obj_1.mean = self.means
        self.cluster_obj_1.variance = self.variances
        self.cluster_obj_1.weight = self.weights
        
        self.cluster_obj_2 = GaussianMixtureModel1D(self.X, 2)
        self.cluster_obj_2.mean = self.means
        self.cluster_obj_2.variance = self.variances
        self.cluster_obj_2.weight = self.weights
        
        self.cluster_obj_3 = GaussianMixtureModel1D(self.X, 2)
        self.cluster_obj_3.mean = self.means
        self.cluster_obj_3.variance = self.variances
        self.cluster_obj_3.weight = self.weights
        
        self.cluster_obj_4 = GaussianMixtureModel1D(self.X, 2)
        self.cluster_obj_4.mean = self.means
        self.cluster_obj_4.variance = self.variances
        self.cluster_obj_4.weight = self.weights
        
    def test_compute_pdf(self):
        """
        Test compute_pdf function from GaussianMixtureModel1D
        """
        pdf = self.cluster_obj_1.compute_pdf(self.X[0], 1)
        self.assertEqual(round(pdf, 3), 0.038)
        
    def test_compute_pdf_matrix(self):
        """
        Test compute_pdf_matrix function from GaussianMixtureModel1D
        """
        pdf_matrix = self.cluster_obj_2.compute_pdf_matrix()
        self.assertEqual(round(pdf_matrix[0,0], 3), 0.044)
        self.assertEqual(round(pdf_matrix[0,1], 3), 0.038)
        
    def tests_compute_posterior(self):
        """
        Test compute_posterior function from GaussianMixtureModel1D
        """
        pdf_matrix = self.cluster_obj_3.compute_pdf_matrix()
        posterior_matrix = self.cluster_obj_3.compute_posterior(pdf_matrix)
        self.assertEqual(round(posterior_matrix[0,0], 2), 0.37)
        self.assertEqual(round(posterior_matrix[0,1], 2), 0.63)
        
    def test_reestimate_params(self):
        """
        Test reestimate_params function from GaussianMixtureModel1D
        """
        pdf_matrix = self.cluster_obj_4.compute_pdf_matrix()
        posterior_matrix = self.cluster_obj_4.compute_posterior(pdf_matrix)
        self.cluster_obj_4.reestimate_params(posterior_matrix)
        self.assertEqual(round(self.cluster_obj_4.mean[0], 2), 0.24)
        self.assertEqual(round(self.cluster_obj_4.variance[0], 2), 0.02)
        self.assertEqual(round(self.cluster_obj_4.weight[0], 2), 0.13)
        
    
# tests = GMMTester()
# myTests = unittest.TestLoader().loadTestsFromModule(tests)
# unittest.TextTestRunner().run(myTests)