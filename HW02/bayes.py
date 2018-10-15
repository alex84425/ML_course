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
import read_data
import math
import numpy as np

from scipy.stats import multivariate_normal as mvn


def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)


class ob_name:

	def __init__(self):
		self.check_arg(sys.argv[1:])
		if self.args.t==0:
			self.discrete_train()
			self.discrete_test()
		else:
			self.continous_train()
			self.continous_test()
			
		print("")

	def check_arg(self,args=None):
		parser = argparse.ArgumentParser(description='Script to learn basic argparse')

		parser.add_argument(	'--t', 
					help='Toglle option: 0:discrete mode, 1: for continuous mode.(defalut is descrete)',
					type=int,
					default=0)
	
		'''
		parser.add_argument(	'filename' ,type=str, 
					help='Log file Path')
	
	
		'''
	
		self.args = parser.parse_args(args)

	def continous_train(self,smoothing=1e-2):
		
		self.gaussians={} 
		self.priors={} # P(c)

		eprint("continous mode!")
		""" read data"""
		d,data_num=read_data.read("training") 
#		data_num=100
		eprint("train data_num: ",data_num)
		
		labels_list= np.array([ d(i)[0] for i in range(data_num) ]  )
		labels=set(labels_list)

		""" calcuate mean and var for each pixel and label!!"""
		for c in labels:
			img_list= np.array([ d(i)[1] for i in range(data_num) if d(i)[0] ==c  ])
#			print(img_list.mean(axis=0).shape )
			self.gaussians[c] = {
				"mean":img_list.mean(axis=0),
				"var":img_list.var(axis=0)+smoothing 
			}
			self.priors[c]=  len([ ele for ele in labels_list if ele== c])/len(labels_list)
#			print( img_list.shape) 
#			break
		pass

	def continous_predict(self,img):
		log_sum_list=[]
		for c,v in self.gaussians.items():
			mean, var = v["mean"], v["var"]
#			print(mean.shape)
#			print(var.shape)
#			print(mvn.logpdf(X, mean= mean, cov=var) + np.log(self.priors[c]))	

			log_sum=0
			for row in range( len(mean) ):
				for col in range( len(mean[0]) ):
					log_sum+=math.log(	1/ (( math.pi*2*var[row][col])**0.5) + math.exp(-0.5*( (img[row][col]-mean[row][col])**2)/var[row][col] )   )
#					log_sum=math.log(log_sum)
					log_sum+= math.log( self.priors[c] )
			log_sum_list.append( [c,log_sum] )
		log_sum_list.sort(key=lambda x:x[1],reverse=True)
		P=log_sum_list[0][0]
		return P
		

	def continous_test(self):
		""" read  test data"""
		d,data_num=read_data.read("testing") 
		eprint("test data_num: ",data_num )
#		data_num=100
		X = np.array([ d(i)[1] for i in range(data_num)  ])
		Y = np.array([ d(i)[0] for i in range(data_num)  ])

		correct=0
		for i,ele in enumerate(X):
			P=(self.continous_predict(ele))
			if Y[i]==P:
#				print("OK!")
				correct+=1
		print("acc: ", correct/len(X))
			
		
		pass

	def predict(self,img):
		t_img=img
#		print("t_label:",t_label)
		score_list=[]
		for l in range(10):#each label
			count_list=[]
#			print("l:",l)
			for row in range(28):
				for col in range(28):
#					print(t_img[row][col],end=" ")
					pixel_count_l=self.two_d_list_of_d[row][col][l][ (t_img[row][col]>>self.bit_shift ) ]
#					print(   pixel_count_l , end=" ")
					count_list.append(  math.log(pixel_count_l) )
#					print(len(self.two_d_list_of_d[row][col][l] ))
#				print()
			score=sum(count_list)+ math.log(  self.label_count[l] / sum(self.label_count)  )*11
#			print()
			score_list.append( [l,score] )
		score_list.sort(key=lambda x:x[1],reverse=True)
#		print(score_list)
		return score_list[0]
		
		pass

	def discrete_train(self):
		self.bit_shift=3

		d,data_num=read_data.read("training") 
		""" init dicttionary"""
		count_d={}
		
		def return_0_9_d(d):
			for i in range(10):
#				d[i]=[1 for j in range( 256/(2**self.bit_shift)  )]
				d[i]=[1 for j in range(32 )]
			return d

		self.two_d_list_of_d = [  [ return_0_9_d( {}) for ele in range(28)  ] for ele in range(28) ]
#		input(self.two_d_list_of_d)
	
		self.label_count=[0 for i in range(10)]
#		data_num=1000	
		eprint("loading data "+str(data_num)+", it will be late!")
		for i in range( data_num ):
			""" [0] in label, [1] is image """
			label=d(i)[0]
			self.label_count[label]+=1
			img=d(i)[1]
#			print("data:", i)
#			print("label:", label)
			for row,p in enumerate(img):
#				print([ ele>>self.bit_shift for ele in p])
				for col,v in  enumerate([ ele>>self.bit_shift for ele in p]):
					self.two_d_list_of_d[row][col][label][v]+=1
			

		print(self.label_count)	



	def discrete_test(self):
		d,data_num=read_data.read("testing") 
		correct=0
		eprint("start training!")
		eprint("train data_num: ",data_num)
		""" test """
		#data_num=1000
		for i in range(data_num):
			t_label=d(i)[0]
			t_img=d(i)[1]
#			print("i:",i)
#			print(self.predict(t_img)[0] )
#			print("t_label:",t_label)
			if t_label==self.predict(t_img)[0]:
				correct+=1
#			input()
		print("accurcy:",correct/data_num)
		pass


if __name__ == '__main__':



		pass

if __name__ == '__main__':

	ob=ob_name()

