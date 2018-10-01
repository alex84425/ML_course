import numpy as np
import sys
from numpy.linalg import inv
 
 
def LU_deco_inverse(m):
	dim = m.shape[0]
	E = np.mat(np.eye(dim))
	L = np.mat(np.eye(dim))
	U = m.copy()
	'''
	print("E:\n",E)
	print("L:\n",L)
	print("U:\n",U)
	'''
	for i in range(dim):
		if abs(m[i,i]) < 1e-8:
			print("zero pivot encoUnted")
			sys.exit()
		# 上面L=m.copy()时用这个，然后我们将其改进先使其初始为单位阵
		# L[:i,i] = 0
		# L[i:dim,i] = U[i:dim,i] / U[i,i]
		L[i+1:,i] = U[i+1:,i] / U[i,i]
 
		# E[i+1:dim,i+1:dim] = E[i+1:dim,i+1:dim] - L[i+1:dim,i]*E[i,i+1:dim]
		# 行变换应该是整个一行的变，而不是上面写的变部分，另E的变换一定要在U之前。
		# 这里还将dim去掉因为意思就是从j+1到最后一个元素，可省略dim看起来没那么晕。
		E[i+1:,:] = E[i+1:,:] - L[i+1:,i]*E[i,:]
		U[i+1:,:] = U[i+1:,:] - L[i+1:,i]*U[i,:]
		# U[i+1:dim,:i+1] = 0
		# U[i+1:dim,i+1:dim] = U[i+1:dim,i+1:dim] - L[i+1:dim,i]*U[i,i+1:dim]
		# 上面这个这样写不划算采用上上面那句代替这俩
 
	print("\nLU分解后的L,U矩阵:")

	
	print("A=\n",m)
	print("L=\n",L)
	print("U=\n",U)
	print("将A化为上三角阵U后随之变换的E矩阵:")
	print("E ( L_inv ) =\n",E)
	print("E_inv( which is L )=\n",inv(E))
	print("LU:\n",L.dot(U))
	input()
 
	# 普通从最后一行开始消去该列的for循环
	# U = U.copy()
	# for i in range(dim-1,-1,-1):
	# 	E[i,:] = E[i,:]/U[i,i]
	# 	U[i,:] = U[i,:]/U[i,i]
	# 	for j in range(i-1,-1,-1):
	# 		E[j,:] = E[j,:] - U[j,i]*E[i,:]
	# 		U[j,:] = U[j,:] - U[j,i]*U[i,:]
 
	# 写成向量形式
	# U = U.copy()
	# for i in range(dim-1,-1,-1):
	# 	E[i,:] = E[i,:]/U[i,i]
	# 	U[i,:] = U[i,:]/U[i,i]
 
	# 	E[i-1:-1:-1,:] = E[i-1:-1:-1,:] - U[i-1:-1:-1,i]*E[i,:]
	# 	U[i-1:-1:-1,:] = U[i-1:-1:-1,:] - U[i-1:-1:-1,i]*U[i,:]
 
	# 通过观察做行变换的过程中发现的规律，比上面注释掉的方法更简单
	E1 = np.mat(np.eye(dim)) # 这个E1用来求U的逆
	for i in range(dim-1,-1,-1):
		# 对角元单位化
		E[i,:] = E[i,:]/U[i,i]
		E1[i,:] = E1[i,:]/U[i,i]
		U[i,:] = U[i,:]/U[i,i]
 
		E[:i,:] = E[:i,:] - U[:i,i]*E[i,:]
		E1[:i,:] = E1[:i,:] - U[:i,i]*E1[i,:]
		U[:i,:] = U[:i,:] - U[:i,i]*U[i,:] # r_j = m_ji - r_j*r_i
 
	print("\n将上三角阵U变为单位阵后的U和随之变换后的E分别为:")
	print("U=",U)
	print("E=",E)
	print("使用系统自带的求inverse方法得到的逆为:")
	print("m_inv=\n",m.I)
	print("\nU的逆E1为:")
	print("E1=",E1)
 
	# 当然，我们还可以来求一下下三角阵L的逆
	E2 = np.mat(np.eye(dim))
	for i in range(dim):
		# 因为这里对角元已经是1了就不做对角元单位化这部了
		E2[i+1:,:] = E2[i+1:,:] - L[i+1:,i]*E2[i,:]
		L[i+1:,:] = L[i+1:,:] - L[i+1:,i]*U[i,:]
 
	print("\n将下三角阵L变为单位阵后的L和随之变换后的E2分别为:")
	print("L=",L)
	print("E2=",E2)
 
	print("\n由A=LU,得A逆=U的逆*L的逆")
	print("U的逆E1*L的逆E2=\n",E1*E2)
 
 
 
 
if __name__ == "__main__":
	"""  problem example"""
#	A= [[2, -3, 4, 2], [6, -9, 12, 5], [4, -5, 10, 5], [2, 2, 11, 9]]
	""" ok example"""
	A=  [ [7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6] ]


#	A = np.mat([[1.,1,1],[1,2,3],[1,5,1]])
	A=np.mat( A )
	A_dim = A.shape[0]
 
	LU_deco_inverse(A)


