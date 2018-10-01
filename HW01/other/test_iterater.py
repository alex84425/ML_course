class Backwards:
	def __init__(self, val):
		self.val = val

		self.i=0
		self.n = len(val)
 
	def __iter__(self):
		return self
 
	def __next__(self):
		# We're done
		if self.i >=  self.n :
			raise StopIteration
 
		self.i += 1
		return self.val[self.i-1]

A=Backwards([1,2,3])
for x in A:
	print(x)
for x in A:
	print(x)
