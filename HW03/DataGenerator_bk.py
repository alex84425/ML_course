from __future__ import print_function
import argparse
import sys
import os
import re
import datetime
import operator
# Fixing random state for reproducibility
import argparse
import sys
import numpy as np
import gzip
from Bio import SeqIO

#from scipy.special import erf
from mpmath import *
import random

from HW01 import matrix

from numpy.random import uniform
from numpy import log, exp, array, matrix
from numpy.linalg import norm, pinv
from scipy.stats import multivariate_normal as mvn
from scipy import stats
import matplotlib.pyplot as plt



def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)



class generator_a:
	def __init__(self,mean,var):
		self.mean=mean
		self.var=var
		pass

	def cal(self):
		var=self.var
		mean=self.mean
		y = random.uniform(0, 1)
		return (erfinv(2*y-1)*var*(2**0.5)+mean)

		'''
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
		'''



class base_function:
	def __init__(self,w):
		self.w_list=[ float(ele) for ele in w.split(",") ]
		
		self.base_num= len(self.w_list)
		
	def cal(self,x):
		l = self.w_list[::-1]
		return sum([ (x**i)*( l[i] ) for i in range(self.base_num) ])
		
		pass



class polynomial_basis_linear_model(base_function):
	def __init__(self,n,w,mean,var):
		self.g = generator_a(mean, var )
		super().__init__(w)

	def cal(self,x):
		return super().cal(x)+self.g.cal()


class HW03:

	def __init__(self):
		self.check_arg(sys.argv[1:])
		self.mean=self.args.m
		self.var=self.args.v
		
#		self.generator_a(self.mean,self.var)
#		self.loop()
		self.main_control()


		pass

	def check_arg(self,args=None):
		""" for 1.a"""
		parser = argparse.ArgumentParser(description='python DataGenerator.py --m 0  --v 0	  --n 2 --a 1 --w "1,1,2"')
	
		parser.add_argument(	'--m' ,type=float, default=0,
					help='input mean')
		parser.add_argument(	'--v' ,type=float, default=1,
					help='input var')
	
		""" for 1.b """
		parser.add_argument(	'--n' ,type=int, default=2,
					help='input base num')
		parser.add_argument(	'--a' ,type=float, default=1,
					help='input var')
		parser.add_argument(	'--w' ,type=str, default=1,
					help='list of number seperated by \",\" ')

		""" for 3."""
		parser.add_argument(	'--b' ,type=float, default=1,
					help='the b, b^-1*I   \",\" ')

		self.args = parser.parse_args(args)
		pass


	def loop(self):
		g = generator_a(self.mean,self.var)
		l=[ g.cal()  for ele in range(1000)]
		print (sum(l) / float(len(l))	)





	def main_control(self):
		#print()

		""" for 1.b """
		p_g=polynomial_basis_linear_model(self.args.n,  self.args.w, 0, self.args.a)
#		print(p_g.cal(3))

		""" for 2. """
		'''
		g =generator_a(self.mean, self.var)
		Sum=0
		Sum_sq=0
		count=0
		for count in range(1,100):
			x=g.cal()	
			Sum += x		
			Sum_sq += x**2
			if count>=2:
				Var = (Sum_sq -((Sum**2) / count)) /(count-1)
#				print(Sum_sq)
#				print(Sum**2)
#				print((Sum**2)/2)
				print( "x: {}, mean: {}, Var: {}".format(x, Sum/count, Var))
			else:
				print("first x:{} ".format(x))
		'''

		""" for 3."""

		prior_u= 0			
		prior_var= 1			
 
		for count in range(1,100):
			x = random.uniform(-10, 10)
			y = p_g.cal(x)
			print( "x:{}, y:{}".format(x,y) )
			M = matrix( [[1,2],[3,4]])			
			print(M )

			input()
		pass


class FullyBaysianRegression():
	def __init__(self, mu=np.zeros(3), sigma=100*np.identity(3), base=3, a=1, b=1):
		self.prior = mvn(mu, sigma)
		self.posterior = self.prior

		self.alpha = a  # real-data noise precision
		self.beta = b   #

		self.cov = matrix(sigma)

		""" initial mean is 0"""
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
		print("x:{} ,phi:{}".format(x,phi))
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


def baysian_linear_regression(precision, n, sigma, w):
	#n = 3
	#sigma = 0.01
	#w = [-0.5, 0.1, 0.03]
	if len(w) != n:
		print("The Dimension of W is not consistant with n!!")
		return False

	""" 100*I """
	S = 100 * np.identity(n)
	m = FullyBaysianRegression(mu=np.zeros(n), sigma=S, base=n, a=n, b=precision)


	while True:
		p_g=polynomial_basis_linear_model(n,  "1,0,0", 0, sigma)
		x = random.uniform(-10,10)
		y =p_g.cal(x)
#		x, y = polynomial_model_data_generator(n, sigma, w)
		m.update_prior(x, y)

		if (norm(m.posterior.mean - m.prior.mean) - 0) < 0.00001:
			break

	return m





if __name__ == '__main__':

#	ob=HW03()
#	ob.check_arg(sys.argv[1:])
	n = 3
	sigma = 0.01
	w = [-0.5, 0.1, 0.03]
	m = baysian_linear_regression(1, n, sigma, w)
	m.draw_model(100)



