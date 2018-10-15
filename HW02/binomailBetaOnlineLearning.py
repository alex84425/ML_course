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


def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)


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
		h=0
		for ele in self.seq_list:
			"""1 is head !  """
			print("new sequence input: :",ele)
			h += len(  [ c for c in ele if str(c)=="1"  ] )
			""" Calculate Likelihood """
			MLE= h / len(ele)
			print("Likelihood: {} ".format(MLE))
			""" Calculate Prior """
			prior =  (a-1)/(a+b-2)
			print("Prior: {} ".format(prior))
			""" Calculate Posterior """
			posterior = MLE * prior
			print("Posterior: {} ".format(posterior))
			""" Calculate new parameters 'a' and 'b' """
			a +=  h
			b += len(ele) - h
#			print( "New shape parameters for Beta Posterior:\n a: {}\n b: {}\n posterior MLE: {} \n".format(a ,b , (a-1)/(a+b-2) ) )
			print( "New shape parameters for Beta Posterior:\n a: {}\n b: {}\n ".format(a ,b , (a-1)/(a+b-2) ) )
			h=0
				
			

		pass
	

if __name__ == '__main__':

	ob=ob_name()


