import numpy as np
"""a = [('a',20),('b',12),('c',9),('d',9)]
print(a)
b = max(a,key=a.count)
a=a[:1]
print(a)
for j in range(0,len(gradients)):
			self.b[j] = self.b[j] - (self.lr*gradients[j])
			for k in range(0,len(self.w)):
				self.w[k][j] = self.w[k][j]
				
transposeInput = self.inputs.T
		transposeWeights = self.w.T
		newWieghts = np.dot(transposeInput,gradients.T)
		newGradient = np.dot(self.w,gradients)
		#print(newGradient)
		self.w = self.w - (self.lr  * newWieghts)
		for grad in gradients:
			self.b = self.b - (self.lr * grad[0])
		#print(self.b)
		#print(self.w)
		#print()
		
		
		
for i in range(0,len(self.b)):
			weights = self.w[:,i]
			neuronValue = np.sum(input*weights) + self.b[i]
			self.neuronValues.append(neuronValue)
		return np.asarray(self.neuronValues)"""
w1 = np.random.normal(0, .1, size=(5, 2))
w2 = np.random.normal(0, .1, size=(10,1))
b1 = np.random.normal(0, .1, size=(1,10))
b2 = np.random.normal(0, .1, size=(1,1))
w5 = np.ones((5,2))
print(w1)
print((0.01*w1))
print()
print((0.01*w1)-w5)
a = np.array([(1,"a"),(2,"b"),(3,"c")])
