import numpy as np
import math


class State:

	def __init__(self, costMatrix, lowerBound, depth, backpointers):
		self.costMatrix = costMatrix
		self.lowerBound = lowerBound
		self.depth = depth
		self.backpointers = backpointers

	def __lt__(self, other):
		return self.weight() < other.weight()

	# lower priority is better
	def weight(self):
		scaledDepth = self.depth / 25
		return self.lowerBound / (1 + scaledDepth)

	def reduceMatrix(self, costMatrix):
		sum = 0

		indicesOfMins = costMatrix.argmin(1)
		for r in range(len(costMatrix)):
			minVal = costMatrix[r][indicesOfMins[r]]
			if minVal != np.inf:
				sum += minVal
				costMatrix[r] -= minVal

		indicesOfMins = costMatrix.argmin(0)
		for c in range(len(costMatrix)):
			minVal = costMatrix[indicesOfMins[c]][c]
			if minVal != np.inf:
				sum += minVal
				costMatrix[:,c] -= minVal

		return sum

	def setSourceAndDest(self, source, dest, costMatrix):
		costMatrix[source].fill(np.inf)
		costMatrix[:, dest].fill(np.inf)
		costMatrix[dest][source] = np.inf

	def genChild(self, dest):
		newBound = self.lowerBound + self.costMatrix[self.backpointers[-1]][dest]
		newMatrix = self.costMatrix.copy()
		self.setSourceAndDest(self.backpointers[-1], dest, newMatrix)
		newBound += self.reduceMatrix(newMatrix)
		newBackpointers = self.backpointers.copy()
		newBackpointers.append(dest)
		return State(newMatrix, newBound, self.depth + 1, newBackpointers)
