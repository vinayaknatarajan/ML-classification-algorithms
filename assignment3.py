import numpy as np
import math

class KNN:
	def __init__(self, k):
		#KNN state here
		#Feel free to add methods
		self.k = k 

	def distance(self, featureA, featureB):
		   diffs = (featureA - featureB)**2
		   return np.sqrt(diffs.sum())

	def train(self, X, y):
		self.xTrain = X
		self.yTrain = y
		return 0		   

	def predict(self, X):
		kNeighbours = list()
		for testData in X:
			neighboursList = list()
			for i in range (0,len(self.xTrain)):
				eucDistance = self.distance(self.xTrain[i],testData)
				distTuple = (self.xTrain[i],self.yTrain[i],eucDistance)
				neighboursList.append(distTuple)
			
			neighboursList.sort(key = lambda x: x[2])
			neighboursList = neighboursList[:self.k]
			kNeighbours.append(neighboursList)
		
		#print(kNeighbours[1],"start here") 
		#Return array of predictions where there is one prediction for each set of features
		solutions = list()
		for neighbours in kNeighbours:
			predictions = [n[1] for n in neighbours]
			label = max(predictions, key = predictions.count)
			solutions.append(label)
		
		#print(solutions)
		#for testValues in X:
				
		
		return np.asarray(solutions)
        
        
            

        

class ID3:
	def __init__(self, nbins, data_range):
		#Decision tree state here
		#Feel free to add methods
		self.bin_size = nbins
		self.range = data_range
		self.counter = 1

	def preprocess(self, data):
		#Our dataset only has continuous data
		norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
		categorical_data = np.floor(self.bin_size*norm_data).astype(int)
		return categorical_data

	def train(self, X, y):
		#training logic here
		#input is array of features and labels
		uniqY,count = np.unique(y,return_counts =True)
		ind = count.argmax()
		self.default = uniqY[ind]
		categorical_data = self.preprocess(X)
		categories = list(range(len(categorical_data[1])))
		categorical_data = np.insert(categorical_data,0,np.asarray(categories),axis=0)
		y = np.insert(y,0,-1)
		self.root = Node()
		self.createTree(categorical_data,y,self.root)
		#print(root.attribute)
		#print(categorical_data)

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		categorical_data = self.preprocess(X)
		solutions = list()
		self.flag =False
		for row in categorical_data:
			self.flag=False
			node = self.root
			#count =0
			#print(node.attribute)
			while node.isLeaf is False:
				value = row[node.attribute]
				#print(node.attribute)
				#print("aaa",value)
				sublist = list()	
				for child in node.children:
					#print(child)
					sublist.append(child[0])
					if (child[0] == value):
						node = child[1]
				if (value not in sublist):
					self.flag =True
					break
			if self.flag is True:
				solutions.append(self.default)
			else:
				solutions.append(node.leafValue)
			
		#print(solutions)
		
		return np.asarray(solutions)
	
	
	def createTree(self,X,y,node):
		if(len(np.unique(y[1:]))==1):
			#print("done")
			node.isLeaf=True
			node.leafValue = y[1]
		else:
			entropy = self.calcEntropy(y[1:])
			infoGain = list()
			y1=y[1:]
			for i in range (0,len(X[1])):
				attr = [attr[i] for attr in X[1:]]
				uniqueLabels,count = np.unique(attr,return_counts = True)
				#print(len(attr))
				entropy2 = 0.0
				for j in range(0,len(uniqueLabels)):
					y2 = y1[np.where(attr == uniqueLabels[j])]
					entropy2 = entropy2 + (count[j]/(len(X)-1))*self.calcEntropy(y2)
				infoGain.append(entropy-entropy2)
			#print(infoGain)
			#try:
			node.attribute = 	X[0][infoGain.index(max(infoGain))]
			#except:
			node.curIndex = infoGain.index(max(infoGain))
			splitAttr = [splitAttr[node.curIndex] for splitAttr in X[1:]]
			splitLabel,count1 = np.unique(splitAttr,return_counts = True)
			for k in range(0,len(splitLabel)):
				xNew = list()
				yNew = list()
				#xNew = X[:1]
				#yNew = y[:1]
				for z in range(1,len(X)):
					if(X[z][node.curIndex] == splitLabel[k]):
						xNew.append(X[z])
						yNew.append(y[z])
						
				xNew = np.asarray(xNew)
				yNew = np.asarray(yNew)
				xNew = np.insert(xNew,0,X[0],axis=0)
				yNew = np.insert(yNew,0,-1)
				xNew = np.delete(xNew,np.s_[node.curIndex:node.curIndex+1],axis=1)
				if(len(xNew[0]) == 1):
					#print("yeah")
					node.isLeaf=True
					node.leafValue = y[1]
				else:
				
					childNode = Node()
					childPointer = (splitLabel[k],childNode)
					node.children.append(childPointer)
					
					self.createTree(xNew,yNew,childNode)
		
	
	def calcEntropy(self,y):
		uniqueLabels,count = np.unique(y,return_counts = True)
		if(len(uniqueLabels) == 1):
			return 0
		else:
			return sum([((-p/len(y-1)) * np.log2(p/len(y-1))) for p in count])
			
		
class Node():
	
   def __init__(self):
	   self.attribute = None
	   self.children = list()
	   self.isLeaf = False
	   self.leafValue = None
	   self.curIndex = None 



class Perceptron:
	def __init__(self, w, b, lr):
		#Perceptron state here, input initial weight matrix
		#Feel free to add methods
		self.lr = lr
		self.w = w
		self.b = b

	def train(self, X, y, steps):
		#training logic here
		#input is array of features and labels
		self.xTrain = X
		self.yTrain = y
		for s in range(0,steps):
			#for i in range(0,len(X)):
				#print(trainData)
				#print(self.w)
				#print(X[i] * self.w)
			i = s % y.size
			neuronValue = np.sum(X[i] * self.w) + self.b
			if neuronValue > 0:
				prediction = 1
			else:
				prediction = 0
			if y[i] == prediction:
				continue
			else:
				offset = y[i] - prediction
				self.b = self.b + offset*self.lr
				for j in range(0,len(self.w)):
					self.w[j] = self.w[j] + (self.lr*offset*X[i][j]) 
			
			#print(neuronValue)
		return 0
		
		
		
		None

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		solutions = list()
		for testData in X:
			neuronValue = np.sum(testData * self.w) + self.b
			if neuronValue[0] > 0:
				solutions.append(1)
			else:
				solutions.append(0)
		
		return np.asarray(solutions)



class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return -2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			#print(pred)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			#print(pred)
			pred = self.a2.forward(pred)
			#print(pred)
			loss = self.MSE(pred, yi) 
			#print(loss)

			grad = self.MSEGrad(pred, yi)
			
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			#print(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)
			
			
			

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		a = np.ravel(pred)
		#print(a)
		return np.ravel(pred)


class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b
		self.inputs =None

	def forward(self, input):
		#Write forward pass here
		self.inputs=input
		
		return (np.dot(input,self.w) + self.b)

	def backward(self, gradients):
		#Write backward pass here
		newWeight = np.dot(self.inputs.T,gradients)
		returnGradients = np.dot(gradients,self.w.T)
		self.w = self.w - (self.lr*newWeight)
		self.b = self.b - (self.lr*gradients)
		return returnGradients	

class Sigmoid:

	def __init__(self):
		None

	def forward(self, input):
		#Write forward pass here
		self.input = 1/(1+np.exp(-input))
		#print(self.input)
		return 1/(1+ np.exp(-input))

	def backward(self, gradients):
		#Write backward pass here
		
		#x=np.flip(gradients*(self.input*(1.0-self.input)),1)
		return gradients*(self.input*(1.0-self.input))