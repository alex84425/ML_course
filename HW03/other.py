
# coding: utf-8

import numpy as np
from numpy.random import uniform
from numpy import log, exp, array, matrix
from numpy.linalg import norm, pinv
from scipy.stats import multivariate_normal as mvn
from scipy import stats
import matplotlib.pyplot as plt


class FullyBaysianRegression():
    def __init__(self, mu=np.zeros(3), sigma=100*np.identity(3), base=3, a=1, b=1):
        self.prior = mvn(mu, sigma)
        self.posterior = self.prior
        
        self.alpha = a  # real-data noise precision
        self.beta = b   #
        
        self.cov = matrix(sigma)
        self.mean = matrix(mu).T
        self.base = base
        
        self.data = [[],[]]
        
        self.iter = 1
        
    def get_phi(self, x, base):
        if isinstance(x, (int, float)):
            W = matrix(np.ones([1, base]))
        else:
            W = matrix(np.ones([len(x), base]))
        tmp = array(x)
        for i in range(1, base):
            m = matrix(tmp).T
            W[:,i] = m
            tmp *= array(x)
    
        return W
    
    def update_prior(self, x, y):
        # Use posterior as new prior
        self.prior = self.posterior
    
        # Create design matrix
        phi = self.get_phi(x, self.base)
        
        # Update mean and covariance
        if self.iter == 0:
            new_precision = self.alpha*phi.T*phi + self.beta*np.identity(self.base)
            self.cov = matrix(pinv(new_precision))
            self.mean = self.cov*self.alpha*phi.T*y
        else:
            new_precision = self.alpha*phi.T*phi + pinv(self.prior.cov)
            self.cov = matrix(pinv(new_precision))
            self.mean = self.cov*(self.alpha*phi.T*y + pinv(self.prior.cov)*self.mean)

        new_mean = self.mean.reshape(phi.shape)
        new_mean = array(new_mean)
    
        self.posterior = mvn(new_mean[0], self.cov)
        self.iter += 1
        self.data[0].append(x)
        self.data[1].append(y)
        
    def predicitve_distribution(self, x):
        # Create design matrix
        phi = self.get_phi(x, self.base)

        mu = phi*self.mean
        cov = 1/self.alpha + phi*self.cov*phi.T
        
        return mvn(mu, cov)
    
    def linear_model(self, w, N=100, v_max=10, v_min=-10):
        x = np.linspace(v_min, v_max, N)
        phi = self.get_phi(x, len(w))
        
        return x, phi*matrix(w).T
    
    def draw_model(self, n):
        draws = stats.multivariate_normal.rvs(self.posterior.mean, self.cov, n)
        for w in draws:
            x, y = self.linear_model(w)
            plt.plot(x, y, alpha=0.1, c='b')
        plt.scatter(self.data[0], self.data[1], facecolors='none', edgecolors='r')

def uni_gaussian_data_generator(mu=0, var=1):
    # Marsaglia polar method
    while True:
        u, v = uniform(-1,1,2)
        s = u**2 + v**2
        
        # Reject Method
        if s >= 1 or s == 0:
            continue
        else:
            break
    
    X = u * np.sqrt(-2*log(s)/s)
        
    return mu + np.sqrt(var)*X

def polynomial_model_data_generator(n, sigma, W):
    if len(W) != n:
        print("The Dimension of W is not consistant with n!!")
        return False
    
    y = W[0]
    x = uniform(-10, 10)
    tmp = x
    for i in range(1, n):
        y += tmp*W[i]
        tmp *= x
    y += uni_gaussian_data_generator(0, sigma)
    
    return x, y

def sequential_estimate_mean_and_variance(m, s, Sum=0, Sumsq=0):
    n = 1
    while True:
        datum = uni_gaussian_data_generator(m, s)
        Sum += datum
        Sumsq += datum**2
        mean = Sum / n
        if n == 1:
            sigmasq = 0
        else:
            sigmasq = (Sumsq - (Sum * Sum) / n) / (n - 1)

        print("N={0}".format(n))
        print("New datum:", datum)
        print("Mean={0}, Var={1}\n".format(mean, sigmasq))

        tolerance = 1e-2
        if abs(mean - m) >= tolerance or abs(sigmasq - s) >= tolerance:
            n += 1
        else:
            break
            
def baysian_linear_regression(precision, n, sigma, w):
    if len(w) != n:
        print("The Dimension of W is not consistant with n!!")
        return False
    
    S = 100 * np.identity(n)
    m = FullyBaysianRegression(mu=np.zeros(n), sigma=S, base=n, a=n, b=precision)
    
    while True:
        x, y = polynomial_model_data_generator(n, sigma, w)
        m.update_prior(x, y)
        
        if (norm(m.posterior.mean - m.prior.mean) - 0) < 0.00001:
            break
            
    return m



# Setup parameter
n = 3
sigma = 0.01
w = [-0.5, 0.1, 0.03]


m = baysian_linear_regression(1, n, sigma, w)

#print("New data : [{}, {}]".format(data_x, data_y))
#print("Posterior Mean:", m.posterior.mean)
#print("Posterior Covariance:\n", m.posterior.cov)

# Draw Model
m.draw_model(100)
