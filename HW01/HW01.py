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
import matplotlib.pyplot as plt
import numpy as np

import copy

import random


class exponential:
	
	def __init__(self,in_list):
		self.para=in_list
		pass

	def cal_y(self,x):
		y=0
		for i,ele in enumerate(self.para[::-1]):
			tmp=x**i
			y+=tmp*ele
		return  y

	def  __str__(self):
		return( "["+( ", ".join( [str(x) for x in self.para  ]  ))+"]"  )		




def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)


def T(m):
	return list(zip(*m)) 
#	return ([[row[i] for row in m] for i in range( len(m) )]  )

'''
class Iter(type):
	def __iter__(cls):
		return iter(cls.l)
'''

class matrix():
	def __init__(self,matrix):
		self.m=matrix
		#for ele in matrix:
		#	self.m.append(ele)
		self.i=0
		self.n=len(self.m)

	def __str__(self):
		#print(	 [  (",".join(  [str(ele) for ele in ele ]))  for ele in self.m ]	)
		str_tmp=( "],\n[".join(   [  (", ".join(  [str(ele) for ele in ele ]))  for ele in self.m ]  )  )
		str_tmp="matrix:\n"+"[["+str_tmp+"]]"
		return  str_tmp

	def g(self):
		return self.m

	def __sub__(self,N):
		for i in range(len(self.m)):
			for j in range(len(self.m[0])):
				self.m[i][j]-=N.g()[i][j]
		return self

	def __add__(self,N):
		for i in range(len(self.m)):
			for j in range(len(self.m[0])):
				self.m[i][j]+=N.g()[i][j]
		return self


	def mul(self,n):
		for i in range(len(self.m)):
			for j in range(len(self.m[0])):
				self.m[i][j]*=2
		return self
				

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
			A_inv=A_inv+self.solve__Ax_b(L,U,list(zip(*[ele])) )
	#   return  list( zip(*A_inv))
		return  matrix(list( zip(*A_inv)))

	def T(self ):
		m=self.m
		#return	matrix([[row[i] for row in m] for i in range( len(m) )]  )
		tmp= list(zip(*m))  
		two_d_list=[ list(ele) for ele in tmp ]
		return matrix ( two_d_list  )
#		return matrix (list(zip(*m)) )

	def __len__(self):
		return len(self.m)

	def __getitem__(self,i):
		return self.m[i]

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

	def solve__Ax_b(self,L,U,b):
		""" calculate y"""
		y=[]
		for i in range (len(L) ):
			tmp=b[i][0]
			y.append( (tmp-sum ([ ele*y[i] for i,ele in enumerate(L[i][:i]) if i>=0  ] ))/L[i][i]   )
		y=[y]
		y=( list(zip(*y))  )
	#	print(y)
		""" calculate x"""
		x=[]
		for i in range (len(L) ):
			tmp=y[ -i-1 ] [0]
			x.append( (tmp-sum ([ ele*x[-i-1] for i,ele in enumerate(U[-i-1][ len(U)-i:  ])   ] ))/U[ -i-1][-i-1]   )
		x=x[::-1]
		x=[x]
		return x
		pass

	
		

class HW:

	def __init__(self):
		self.check_arg(sys.argv[1:])
		self.input_to_matrix()
		self.LSE()

		self.newton_method()
		self.calulate_loss( self.LSE_solution)
		self.calulate_loss( self.newton_solution)
		self.draw(self.LSE_solution,"LSE")
		self.draw(self.newton_solution,"newton")
		pass

	def check_arg(self,args=None):
		parser = argparse.ArgumentParser(description='python HW01.py  --i input_1_2_3_4.txt  --b 4 --l 0')

		parser.add_argument(	'--i' ,type=str, 
					default="input_auto.txt",
					help=' input -file')
		parser.add_argument(	'--l' ,type=float, 
					default=1 ,
					help='input lambda')
		parser.add_argument(	'--b' ,type=int, 
					default=3 ,#mean ax^2+bx^1+x=y
					help='input bases')
		self.args = parser.parse_args(args)

	def basis_function(self,x):
		b=self.args.b
		return [ x**p for p in range(b) ][::-1]

	def input_to_matrix(self):
	
		x_list=[]
		y_list=[]
		A=[]
		b=[]
		with open(self.args.i) as f:
			content=f.readlines()
			for line in content:
				line=line.strip().replace(" ","")
				if line[0]!="#":
					item=line.split(",")
					#print(self.basis_function(int(item[0])  ))
					A.append(  (self.basis_function(int(item[0])  ))  )
					b.append( [int(item[1] )] )
					x_list.append( int(item[0]))
					y_list.append( int(item[1]))
				else:
					self.true_ans=line
		
		""" assign two d list to matrix object"""	
		A=matrix( A  )
		b=matrix( b  )
		self.A=A
		self.b=b
		self.x_list=x_list
		self.y_list=y_list

	def LSE(self):
		A=self.A
		b=self.b
		ATA=matrix( A.T()*A  )
		""" add lambda l=1 (defalut)"""
		ATA=ATA.add_lambda( self.args.l )
		ATA_inv=ATA.inv()

		ATA_inv_AT=ATA_inv *(A.T() )
		""" (A^T*A)^-1 *A^T *b """
		ATA_inv_AT_b= ATA_inv_AT  *b 
		print("LSE ans:")
		print(ATA_inv_AT_b)
		self.LSE_solution= ATA_inv_AT_b
		pass


	def calulate_loss(self,x):
		print("in calculate loss!")

		""" Ax = b' (x is we calculate  , b' is expect b)"""
		expect=( (self.A*x).T()[0] )
		""" ans = b  """
		ans=( (self.b).T()[0] )
		print("loss:", sum( [ (expect[i]-ans[i])**2   for i in range(len(ans))  ])   )  
		pass

	def newton_method(self):
		A=self.A
		b=self.b
	
		x=matrix( [[ random.randint(0,100) for i in range( len(A[0])) ]])
		x=x.T()
		for i in range(100):
			"""Hession_matrix"""
			H_m= (A.T()*A).mul(2).inv()
	
			"""derivation"""
			d = (A.T()*A*x-(A.T()*b)  ).mul(2)
			"""new x """
			x= x-(H_m*d )		
			'''
			print(H_m*d )		
			print(x)
			print( x-(H_m*d) )
			'''
		print("Newton method ans:")
		print(x)	
		self.newton_solution=x
		pass


	def draw(self,solution,title):
		tmp= sorted([ ele for ele in  zip( self.x_list,self.y_list) ],key=lambda x:x[0])

		self.x_list= [ele[0] for ele in tmp]
		self.y_list= [ele[1] for ele in tmp]
		plt.figure()
		""" input dot or line"""
		plt.plot(self.x_list, self.y_list,"ro") #ro mean dots

		a1=(solution)
#		a2=( self.newton_solution)

		""" """
		ob=exponential( a1.T()[0])
		expect_y=(ob.cal_y( np.array( self.x_list) ))
		plt.plot(self.x_list, expect_y,color="blue") #ro mean dots

		'''
		print(a2.T())		
		ob=exponential( a2.T()[0])
		expect_y=(ob.cal_y( np.array( self.x_list) ))
		plt.plot(self.x_list, expect_y,color="green") #ro mean dots
		'''



		"""for base para """
		flank=10
		plt.xlim((min(self.x_list)-flank, max(self.x_list)+flank ))
		plt.ylim((min(self.y_list)-flank, max(self.y_list)+flank ))
		plt.xlabel('x axis')
		plt.ylabel('y axis')
		plt.title(title)
		plt.show()

		pass


if __name__ == '__main__':
	
#	ex_1=exponential([1,2,3,4])
#	ex_1=exponential([1,2,1])
	ex_1=exponential([1,2,1])

	""" create input file """
	print("input dots:")
	with open("input_auto.txt","w") as f:
		f.write( "#"+str(ex_1)+"\n")
		for i in range(100):
			x=random.randint(-10,10)
			print(  "{},{}".format(x, ex_1.cal_y(x))   )
			f.write(  "{},{}\n".format(x, ex_1.cal_y(x))   )
		
	print()
	ob=HW()
		



