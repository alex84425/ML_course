'''
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
import pprint as pp
import numpy as np
from numpy.linalg import inv
import copy
'''



class matrix():
	def __init__(self,matrix):
		self.m=matrix
		#for ele in matrix:
		#	self.m.append(ele)
		self.i=0
		self.n=len(self.m)
		
		
		pass

	def __str__(self):
		#print(	 [  (",".join(  [str(ele) for ele in ele ]))  for ele in self.m ]	)
		str_tmp=( "],\n[".join(   [  (", ".join(  [str(ele) for ele in ele ]))  for ele in self.m ]  )  )
		str_tmp="matrix:\n"+"[["+str_tmp+"]]"
		return  str_tmp

	def g(self):
		return self.m


	def __mul__(self,N):
		M=self.m
		N_T= list(zip( *(N.g() ) )) 
		return matrix( [[sum(el_m * el_n for el_m, el_n in zip(row_m, col_n)) for col_n in N_T ] for row_m in M] ) 

	def mult_matrix(self,M,N):
		"""Multiply square matrices of same dimension M and N"""
		# Converts N into a list of tuples of columns
		""" transpose """
		N_T= list(zip(*N)) 
		return [[sum(el_m * el_n for el_m, el_n in zip(row_m, col_n)) for col_n in N_T ] for row_m in M]

	def inv(self):
	
		A=self.m
		n=len(A)
		"""b=I """
		b = [[float(i == j) for i in range(n)] for j in range(n)]
	
		"""get L, U without check A is invertible or not """
		A=matrix(A)
		L,U=A.lu_decomposition()
		A_inv=[]
		for ele in b:
			solve__Ax_b(L,U,list(zip(*[ele])) )
			A_inv=A_inv+solve__Ax_b(L,U,list(zip(*[ele])) )
	#   return  list( zip(*A_inv))
		return  matrix(list( zip(*A_inv)))




	def T(self ):
		m=self.m
		return matrix (list(zip(*m)) )
		#return	matrix([[row[i] for row in m] for i in range( len(m) )]  )

	def __len__(self):
		return len(self.m)

	def __getitem__(self,i):
		return self.m[i]

	'''
	def __iter__(self):
		return self

	def __next__(self):
		# We're done
		if self.i >=  self.n :
			raise StopIteration

		self.i += 1
		return self.m[self.i-1]

	'''
	def lu_decomposition(self):
		A=self.m
		"""Performs an LU Decomposition of A (which must be square)
		into PA = LU. The function returns P, L and U."""
		n = len(A)
		# Create zero matrices for L and U
		L = [[0.0] * n for i in range(n)]
		U = [[0.0] * n for i in range(n)]
		# Create the pivot matrix P and the multipled matrix PA
	#   P = pivot_matrix(A)
		#PA = mult_matrix(P, A)
		"""skip checking pivot is zero or not"""
		PA =A
	
		# Perform the LU Decomposition
		for j in range(n):
			# All diagonal entries of L are set to unity
			L[j][j] = 1.0
	
			# LaTeX: u_{ij} = a_{ij} - \sum_{k=1}^{i-1} u_{kj} l_{ik}
			for i in range(j+1):
				s1 = sum(U[k][j] * L[i][k] for k in range(i))
				U[i][j] = PA[i][j] - s1
	
			# LaTeX: l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik} )
			for i in range(j, n):
				s2 = sum(U[k][j] * L[i][k] for k in range(j))
				L[i][j] = (PA[i][j] - s2) / U[j][j]
		return ( L, U)


	def add_lambda(self,l):
		"""l constant """
		n=len(self.m)
		m=self.m
		return matrix( [ [  ((0+l)*float(i ==j))+m[i][j] for i in range(n)] for j in range(n)]  )

		

