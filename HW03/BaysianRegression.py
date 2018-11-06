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
	def __init__(self,mean = 0 ,var = 1):
		self.mean=mean
		self.var=var
		pass

	def cal(self):
		var=self.var
		mean=self.mean
#		y = random.uniform(0, 1)
#		return (erfinv(2*y-1)*var*(2**0.5)+mean)


		""" Marsaglia polar method """
		while True:
			u, v = uniform(-1,1,2)

			s = u**2 + v**2
			# Reject Method
			if s >= 1 or s == 0:
				continue
			else:
				break

		Z = u * np.sqrt(-2*log(s)/s)
		""" X = μ + σZ """
		return mean + np.sqrt(var)*Z



class base_function:
	def __init__(self,w):
	
		if isinstance(w,list):
			self.w_list=w
		else:
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


	def sequential_estimate_mean_and_variance(self):
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
				print( "x: {}, mean: {}, Var: {}".format(x, Sum/count, Var))
			else:
				print("first x:{} ".format(x))

		pass


	def baysian_linear_regression(self, precision, n, sigma, w):
		#n = 3
		#sigma = 0.01
		#w = [-0.5, 0.1, 0.03]
		if len(w) != n:
			print("The Dimension of W is not consistant with n!!")
			return False
	
		""" 100*I , initial random value by user"""
		S = 100 * np.identity(n)
		m = FullyBaysianRegression(mu=np.zeros(n), sigma=S, base=n, a=1/(sigma**2), b=precision)
		#m = FullyBaysianRegression(mu=np.zeros(n), sigma=S, base=n, a=1, b=precision)
	
		while True:
			p_g=polynomial_basis_linear_model(n,  w, 0, sigma)
			x = random.uniform(-10,10)
			y =p_g.cal(x)
	#		x, y = polynomial_model_data_generator(n, sigma, w)
			print("x:{}, y:{}".format(x, y))
			new_mean, cov=m.update_prior(x, y)
	
			#print("mean: {}".format(new_mean))
			#print("cov: {}".format(cov))
			""" stop iter if psterior and prior is too close"""
			if (norm(m.posterior.mean - m.prior.mean) - 0) < 0.00001:
				print("mean: {}".format(new_mean))
				print("cov: {}".format(cov))
				break
	
		return m


	def main_control(self):
		#print()

		""" for 1.a: univariate gaussian data generator  """

		if(0):
			g = generator_a(10,1)
			#print(g.cal())
			with open("100_random_generated.txt","w") as f:
				for i in range(1000):
					f.write( str(float(g.cal()) )+"\n" )
				
			os.system("python ~/small_tool/histogram_for_general.py --i 100_random_generated.txt --bin 0.1")

		""" for 1.b: polynomial basis linear model ("""
		p_g=polynomial_basis_linear_model(self.args.n,  self.args.w, 0, self.args.a)
		x = random.uniform(-10,10)
		#print(p_g.cal(x))
		""" for 2. """
#		self.sequential_estimate_mean_and_variance()

		""" for 3."""
		sigma = self.args.a
		w = [float(ele) for ele in self.args.w.split(",")  ]
		n = len(w)
		#(precision, base_num,sigma, w  )
		m = self.baysian_linear_regression( 1, n, sigma, w) 
		m.draw_model(100)



class FullyBaysianRegression():
	def __init__(self,mu=np.zeros(3), sigma=100*np.identity(3), base=3,a=1,b=1):
		self.prior = mvn(mu,sigma)
		self.posterior = self.prior

		self.alpha = a  # real-data noise precision
		self.beta = b

		self.cov=matrix(sigma)
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
#		print("x:{} ,phi:{}".format(x,phi))
		# Update mean and covariance
		if self.iter == 0:
			""" 
			something should know first:
				cov = sigma = precision_matrix^-1 
				cov^-1 = b *I
			"""

			""" for the first interation"""

			""" ^(precision_matrix)=  (  a * X * X^T ) + b * I """
			new_precision = self.alpha*phi.T*phi + self.beta*np.identity(self.base)
			"""  ^ =  cov^-1 """
			self.cov = matrix(pinv(new_precision))
			""" a * cov  * X^T * y"""
			self.mean = self.cov*self.alpha*phi.T*y
		else:
			""" 
				( a * X * X^T ) + b * I
				b * I = inv( prior.cov )
			"""
			new_precision = self.alpha*phi.T*phi + pinv(self.prior.cov)

			""" 
				 prior.cov  = inv(  b * I )
			"""
			self.cov = matrix(pinv(new_precision))

			""" 
				cov * ( a * X^T * y + S * m)
				S^-1 = prior.cov
			"""
			self.mean = self.cov*(self.alpha*phi.T*y + pinv(self.prior.cov)*self.mean)

		new_mean = self.mean.reshape(phi.shape)
		new_mean = array(new_mean)


		self.posterior = mvn(new_mean[0], self.cov)
		self.iter += 1

#		print("mean: {}".format(new_mean))
#		print("cov: {}".format(self.cov))
		self.data[0].append(x)
		self.data[1].append(y)
		return (new_mean, self.cov)

	def predicitive_distribution(self, x):
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
		for i,w in enumerate(draws):
			x, y = self.linear_model(w)
			plt.plot(x, y, alpha=0.1, c='b')
#			print(i)
		plt.scatter(self.data[0], self.data[1], facecolors='none', edgecolors='r')
		plt.show()




if __name__ == '__main__':

	ob=HW03()
	ob.check_arg(sys.argv[1:])



