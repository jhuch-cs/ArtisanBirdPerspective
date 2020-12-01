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
from threading import Timer


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
				current_city = self.nearest_neighbor(current_city, cities)
				route.append(current_city)
				cities.remove(current_city)
			possible = TSPSolution(route)
			if bssf == None or bssf.cost > possible.cost:
				bssf = possible
				count += 1
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

	def nearest_neighbor(self, city, cities):
		return min(cities, key=lambda x: city.costTo(x))
	
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
		S = [state]
		
		while S:
			if time.time() - start_time > time_allowance:
				break
			P = heapq.heappop(S)
			if P.lowerBound < bssf.cost:
				for i in (set(range(n_cities)) - set(P.backpointers)):
					childState = P.genChild(i)
					total += 1
					if len(childState.backpointers) == n_cities and childState.lowerBound < bssf.cost:
						bssf = TSPSolution([cities[x] for x in childState.backpointers])
						count += 1
					elif childState.lowerBound < bssf.cost:
						heapq.heappush(S, childState)
						maxSize = max(maxSize, len(S))
					else:
						pruned += 1
			else:
				pruned += 1
		
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
		pass
		



