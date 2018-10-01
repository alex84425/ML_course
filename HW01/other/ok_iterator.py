class Iter(type):
	def __iter__(cls):
		return iter(cls.l)


class Person(metaclass=Iter):
#	allPeople = []
	l=[]

	def __init__(self, l):
#		self.allPeople.append(self)
#		self.l=l
		for ele in l:
			self.l.append(ele)
		


if __name__ == '__main__':
	Jeff = Person([1,2,3,4,5])
	Jeff = Person([1,2,3,4,5])

	for person in Person:
		print(person)
	for person in Person:
		print(person)
