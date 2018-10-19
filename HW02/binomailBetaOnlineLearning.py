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
import math
import matplotlib.pyplot as plt



def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)


class Beta_function:
	def __init__(self,a=1,b=1):
		self.a=a
		self.b=b
		pass


	def beta_ditribution(self,x):
		a = self.a
		b = self.b
		'''uses gamma function or inbuilt math.gamma() to compute values of beta function'''
#		beta = math.gamma(a+b)/ ( math.gamma(a)*math.gamma(b) )
		beta = math.exp(-math.lgamma(a) - math.lgamma(b) + math.lgamma(a+b))
		beta*=  (x**(a-1 ))*((1-x)**(b-1))
		return beta

	def pdf(self, data) :
		if isinstance(data, (int, float)) :
			x=data
			return x**(self.a-1)*(1-x)**(self.b-1)*comb(self.a+self.b-1, self.a-1)*self.b

		else :
			for x in data:
				return [x**(self.a-1)*(1-x)**(self.b-1)*comb(self.a+self.b-1, self.a-1)*self.b for x in data  ]
	def mle(self):
		return (self.a -1 ) / (self.b +self.a -2)

	def plot(self,title):

		self.x_list= ( [ i/100 for i in range(100) ] )
		self.y_list= ( [ self.beta_ditribution(i/100) for i in range(100) ] )
#		print( sum([ self.beta_ditribution(i/100) for i in range(100) ])/100 )
		flank = 0
		plt.figure()
		""" input dot or line"""
		plt.plot(self.x_list, self.y_list,"ro") #ro mean dots
		plt.xlim((min(self.x_list)-flank, max(self.x_list)+flank ))
		plt.ylim((min(self.y_list)-flank, max(self.y_list)+flank ))
		plt.ylabel('likelihood')
		plt.xlabel('x')
		plt.title(title)
		plt.show()





		pass



class ob_name:

	def __init__(self):
		self.check_arg(sys.argv[1:])
		self.read_input()
		self.calculate_each_MLE()
		pass

	def check_arg(self,args=None):
		parser = argparse.ArgumentParser(description='Script to learn basic argparse')
	
		parser.add_argument(	'--i' ,type=str, 
					help='input sequence file')
		parser.add_argument(	'--a' ,type=int, 
					default=2,
					help='prior of a (head times)')
		parser.add_argument(	'--b' ,type=int, 
					default=2,
					help='prior of b (tail times)')
		self.args = parser.parse_args(args)
		'''
		parser.add_argument(	'-u', 
					help='Summary failed login log and sort log by user',
					action="store_true")
	
		'''
	def read_input(self):
		self.seq_list=[]
		with open(self.args.i ,"r") as f:
			content=f.readlines()
			for line in content :
				line=line.strip()
				self.seq_list.append(line)


	def calculate_each_MLE(self):
		a = self.args.a 
		b = self.args.b 
		prior = Beta_function(a,b)

		for i, ele in enumerate(self.seq_list):
			"""1 is head !  """
			print ( "prior MLE --> {1} a={2}, b={3} ".format(i, prior.mle(), prior.a , prior.b) )
			prior.plot("prior "+str(i))
			print("new sequence input: :",ele)
			a = len(  [ c for c in ele if str(c)=="1"  ] )
			b = len(ele) -a 
			posterior = Beta_function( prior.a +a, prior.b+b )
			prior = posterior
#			print("data[{0}]: {1}".format(i,  ele ))
			print("MLE --> {}".format(a/(a+b)))
			print ( "posterior MLE --> {1} a={2}, b={3} \n".format(i, posterior.mle(), posterior.a , posterior.b) )

			

		pass
	

if __name__ == '__main__':

	ob=ob_name()


