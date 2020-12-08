#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
from State import *
import heapq
import itertools
import signal
from random import shuffle
from random import randint
from random import choice
from random import randrange
from math import ceil



def handler():
	raise Exception("end of time")

class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		self.greedySolutions = []
		results = {}
		cities = self._scenario.getCities().copy()
		n_cities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		route = []
		start_time = time.time()
		i = 0
		while i < n_cities:
			current_city = cities[i]
			route.append(current_city)
			cities.remove(current_city)
			while len(route) < n_cities:
				current_city = min(cities, key=lambda x: current_city.costTo(x))
				route.append(current_city)
				cities.remove(current_city)
			possible = TSPSolution(route)
			if bssf == None or bssf.cost > possible.cost:
				self.greedySolutions.append(route)
				bssf = possible
				count += 1
			elif possible.cost != math.inf:
				self.greedySolutions.append(route)
			cities = self._scenario.getCities().copy()
			route = []
			i += 1
		end_time = time.time()
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound( self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		n_cities = len(cities)
		count = 0
		pruned = 0
		maxSize = 0
		total = 1
		route = []
		bssf = self.greedy()['soln']
		start_time = time.time()

		costMatrix = self.createInitialMatrix(cities)
		state = State(costMatrix, 0, 1, [0])
		state.lowerBound = state.reduceMatrix(costMatrix)
		states = [state]
		
		while states:
			if time.time() - start_time > time_allowance:
				break
			currentState = heapq.heappop(states)
			if currentState.lowerBound < bssf.cost:
				for i in (set(range(n_cities)) - set(currentState.backpointers)):
					childState = currentState.genChild(i)
					total += 1
					if len(childState.backpointers) == n_cities and childState.lowerBound < bssf.cost:
						bssf = TSPSolution([cities[x] for x in childState.backpointers])
						count += 1
					elif childState.lowerBound < bssf.cost:
						heapq.heappush(states, childState)
						maxSize = max(maxSize, len(states))
					else:
						pruned += 1
			else:
				pruned += 1
		
		pruned += len(states)
		
		end_time = time.time()
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = maxSize
		results['total'] = total
		results['pruned'] = pruned
		return results

	def createInitialMatrix(self, cities):
		n = len(cities)
		costMatrix = np.full((n, n), np.inf)
		for i in range(n):
			for j in range(n):
				costMatrix[i][j] = cities[i].costTo(cities[j])
		return costMatrix






	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
	def fancy( self,time_allowance=60.0 ):
		results = {}
		count = 0
		start_time = time.time()

		self.bssf = (math.inf, math.inf)
		self.greedy()
		self.cities = self._scenario.getCities()
		self.weightedOdds = ['1'] + ['2'] * 2 + ['3'] * 3 + ['4'] * 5

		population = self.initializePopulation()

		numGenerations = 10000
		for i in range(numGenerations):
			PERCENT_PARENTS = 1
			numParents = ceil(len(population) * PERCENT_PARENTS)
			parents = self.select(self.fitness(population), numParents)
			children = [self.crossover(x[0][0], x[1][0]) for x in parents]
			self.mutateAll(children)
			population = self.survive(population, children)

		bssf = self.bssf
		possible = self.highestFitness(population)
		bssf = self.decode(possible[0]) if possible[1] <= bssf[1] else self.decode(bssf[0])

		end_time = time.time()
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results
		
	def initializePopulation(self):
		population = self.getGreedySolutions()
		population.sort(key=lambda x: self.individualFitness(x))

		population = population[:8]
		INITIAL_SIZE = ceil(len(self.cities) / 2) if len(self.cities) < 20 else 10

		while len(population) < INITIAL_SIZE:
			population.append(np.random.permutation(len(self.cities)).tolist())
		return population
		
	def getGreedySolutions(self):
		return [self.encode(x) for x in self.greedySolutions]
	
	def encode(self, route):
		return [city._index for city in route]

	def decode(self, individual):
		cities = self._scenario.getCities()
		return TSPSolution([cities[index] for index in individual])

	def highestFitness(self, population):
		return (population[0], self.individualFitness(population[0]))

	def survive(self, population, children):
		PERCENT_CULLED = 0.3
		numToCull = ceil((len(population) + len(children)) * PERCENT_CULLED)

		population += children

		population.sort(key=lambda individual: self.individualFitness(individual))

		self.cull(population, numToCull)

		return population

	def cull(self, population, numToCull):
		for i in range(numToCull):
			section = choice(self.weightedOdds)
			if section == '4':
				index = randint(int(len(population) * 0.75), len(population) - 1)
			elif section == '3':
				index = randint(int(len(population) / 2), int(len(population) * 0.75))
			elif section == '2':
				index = randint(int(len(population) * 0.25), int(len(population) / 2))
			else:
				index = randint(0, int(len(population) * 0.25))
				if index == 0:
					fitness = self.individualFitness(population[index])
					if fitness < self.bssf[1]:
						self.bssf = (population[0], fitness)
			del population[index]
	
	def fitness(self, population):
		cities = self.cities
		tuple_info = []
		for individual in population:
			total_cost = 0
			for i in range(0, len(individual) - 1):
				total_cost += cities[individual[i]].costTo(cities[individual[i+1]])
			total_cost += cities[individual[-1]].costTo(cities[individual[0]])
			final_info = (individual, total_cost)
			tuple_info.append(final_info)
		tuple_info.sort(key=lambda x: x[1])
		return tuple_info

	def individualFitness(self, individual):
		cities = self.cities
		total_cost = 0
		for i in range(0, len(individual) - 1):
			total_cost += cities[individual[i]].costTo(cities[individual[i+1]])
		total_cost += cities[individual[-1]].costTo(cities[individual[0]])
		return total_cost

	def select(self, population, numParents):
		breedingPopulation = []
		firstQ = int(len(population) * 0.25)
		secondQ = int(len(population) / 2)
		thirdQ = int(len(population) * 0.75)

		for i in range(numParents):
			section = choice(self.weightedOdds)
			if section == '4': # occurs most frequently
				index = randint(0, firstQ)
			elif section == '3':
				index = randint(firstQ, secondQ)
			elif section == '2':
				index = randint(secondQ, thirdQ)	
			else: # occurs least frequently
				index = randint(thirdQ, len(population) - 1)
			breedingPopulation.append(population[index])

		list_of_couples = []

		if len(breedingPopulation) % 2 != 0:
			breedingPopulation.append(breedingPopulation[0])
		list_of_couples = zip(breedingPopulation[::2], breedingPopulation[1::2])

		# Couples = [[(Solution1, Solution1 Fitness), (Solution2, Solution2 Fitness)] , ... , [(SolutionN-1, SolutionN-1 Fitness), (SolutionN, SolutionN Fitness)]]
		return list_of_couples

	# Swaps two random cities in a given path
	# This is O(n), because it requires building a child path
	def mutate(self, parent):
		n = len(parent)

		# Get cities to swap
		index1 = int(random.random() * n)
		index2 = int(random.random() * n)

		parent[index1], parent[index2] = parent[index2], parent[index1]

	def mutateAll(self, population):
		mutatationRate = .3
		numMutants = ceil(mutatationRate * len(population))
		for i in range(numMutants):
			self.mutate(choice(population))
			
	# Breeds two routes by taking a subsection of one parent and
	# appending cities not in that subsection in the order they
	# appear in the second parent
	# This function is O(n), because assembling the child must be
	# O(n)
	def crossover(self, parent1, parent2):
		n = len(parent1)

		# Get subsection
		index1 = int(random.random() * (n+1))
		index2 = int(random.random() * (n+1))

		startIndex = min(index1, index2)
		endIndex = max(index1, index2)

		p1Genes = []
		# This is O(n), since startIndex could be 0 and endIndex n
		for i in range (startIndex, endIndex):
			p1Genes.append(parent1[i])

		# This is O(n), since startIndex could equal endIndex
		p2Genes = [gene for gene in parent2 if gene not in p1Genes]

		# This is O(n), because it assembles the child's route
		childRoute = p2Genes[:startIndex] + p1Genes + p2Genes[startIndex:]

		return childRoute


