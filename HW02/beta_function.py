import math
 
def beta_1(x,a,b):
	 
	'''uses gamma function or inbuilt math.gamma() to compute values of beta function'''
	#beta = math.gamma(a)*math.gamma(b)/math.gamma(a+b) 
	beta = math.gamma(a+b)/ ( math.gamma(a)*math.gamma(b) )
	beta*=	(x**(a-1 ))*((1-x)**(b-1))
	return beta

def beta_origin(a,b):
	beta = math.gamma(a+b)/ ( math.gamma(a)*math.gamma(b) )
	return beta


print(beta_1(0.5,10,10) )
#print( beta_origin(3,3) )
#print( beta_origin(4,4) )
#print( math.gamma(7) )

