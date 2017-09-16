
import numpy as np
from random import randint
from colorama import init, Fore, Style
from metrics import majority_voting_score
from sklearn.externals.joblib import Parallel, delayed


def _get_individual_score(individual, estimators, X, y, classes):
	selected_estimators = [estimator for estimator, isSelected in zip(estimators, individual) if isSelected]
	individual_score = majority_voting_score(X, y, selected_estimators, classes)
	return individual_score

def _get_population_scores(pop_slice, estimators, X, y, classes):
	return [_get_individual_score(individual, estimators, X, y, classes) for individual in pop_slice]

class GeneticOptimizer(object):
	"""
	Class to wrap evolutionary genetic optimization operations
	"""
	def __init__(self, 
				 estimators, 
				 classes, 
				 data, 
				 target, 
				 pop_size=30, 
				 mutation_rate=0.1, 
				 crossover_rate=0.9, 
				 iterations=1000, 
				 n_jobs=1):		
		self.estimators = estimators
		self.classes = classes
		self.X = data
		self.y = target
		self.pop_size = pop_size
		self.mutation_rate = mutation_rate
		self.crossover_rate = crossover_rate
		self.iterations = iterations
		self.n_jobs = n_jobs
		self.__generate_random_population()
		init()

	def __generate_random_population(self):
		print("\nGenerating random population")
		print("Population size = %d" % self.pop_size)
		print("Individual size (genes) = %d" % len(self.estimators))
		self.pop = []
		for i in range(self.pop_size):
			individual = []
			for j in range(len(self.estimators)):
				individual.append(randint(0, 1))
			self.pop.append(individual)

	def __compute_starts(self):
		n_individuals_per_job = (len(self.pop) // self.n_jobs) * np.ones(self.n_jobs, dtype=np.int)
		n_individuals_per_job[:len(self.pop) % self.n_jobs] += 1
		starts = np.cumsum(n_individuals_per_job)
		return ([0] + starts.tolist())

	def __parallel_score_processing(self):
		starts = self.__compute_starts()

		scores = Parallel(
			n_jobs=self.n_jobs, verbose=0)(
			delayed(_get_population_scores)(self.pop[starts[i]:starts[i + 1]],
											self.estimators,
											self.X,
											self.y,
											self.classes) for i in range(self.n_jobs))
		scores = [score for score_list in scores for score in score_list]
		return scores

	def __calculate_fitness_probabilities(self, scores):
		scores_sum = np.sum(scores)
		reproduction_prob = [score/scores_sum for score in scores]
		return reproduction_prob

	def __update_pop_history(self):
		self.pop_history = []
		pop_hashes = [hash(e) for e in [''.join([str(s) for s in x]) for x in self.pop]]
		for pop_hash in pop_hashes:
			if pop_hash not in self.pop_history:
				self.pop_history.append(pop_hash)

	def __crossover(self, pair):
		cut_index = randint(1, len(pair[0])-2)
		individual_1 = pair[0]
		individual_2 = pair[1]
		pair[0] = individual_1[:cut_index] + individual_2[cut_index:]
		pair[1] = individual_2[:cut_index] + individual_1[cut_index:]
		return pair

	def __mutate(self, individual):
		if (np.random.rand() <= self.mutation_rate) or ((np.random.rand() <= 0.9) and (hash(''.join([str(s) for s in individual])) in self.pop_history)):
			n_mutations = randint(0, len(individual) // 4)
			for _ in range(n_mutations):
				index_mutation = randint(0, len(individual)-1)
				individual[index_mutation] ^= 1
		return individual

	def __reproduce_population(self, fitness_prob, sel_sensivity=0.9):
		sorted_pop = [pop for _, pop in sorted(zip(fitness_prob, self.pop))]

		sel_sensivity = 0.85
		a = np.arange(1, len(sorted_pop) + 1)
		sel_prob = [((sel_sensivity - 1) / ((sel_sensivity**len(a)) - (1))) * (sel_sensivity**(len(a)-i)) for i in a]

		new_pop = []
		crossover_pop = []
		for i in range(int(len(self.pop)/2)):
			# pair = [sorted_pop[i], self.pop[np.random.choice(len(self.pop), 1, p=sel_prob)[0]]]
			pair_indexes = np.random.choice(len(sorted_pop), 2, replace=False, p=sel_prob)
			pair = [sorted_pop[pair_indexes[0]], sorted_pop[pair_indexes[1]]]
			pair = self.__crossover(pair)
			crossover_pop.append(pair[0])
			crossover_pop.append(pair[1])
		new_pop = [self.__mutate(individual) for individual in crossover_pop]
		return new_pop

	def __rank_population(self):
		scores = self.__parallel_score_processing()
		sorted_pop = [individual for _, individual in sorted(zip(scores, self.pop), reverse=True)]
		return sorted_pop, scores

	def __calculate_population_stats(self, initial_score, prev_score):
		sorted_pop, scores = self.__rank_population()
		best_individual = sorted_pop[0]
		best_score = _get_individual_score(best_individual, self.estimators, self.X, self.y, self.classes)
		if best_score > prev_score:
			print("Best individual score: " + Fore.GREEN + Style.BRIGHT + "%f" % best_score + Style.RESET_ALL + " (%f%%)" % ((best_score - initial_score) * 100))
		elif best_score < prev_score:
			print("Best individual score: " + Fore.RED + Style.BRIGHT + "%f" % best_score + Style.RESET_ALL + " (%f%%)" % ((best_score - initial_score) * 100))
		else:
			print("Best individual score: " + Fore.CYAN + Style.BRIGHT + "%f" % best_score + Style.RESET_ALL + " (%f%%)" % ((best_score - initial_score) * 100))

		t = 5
		print("Top %d scores:" % t)
		top_individuals = sorted_pop[:t]
		print(["%f" % ((_get_individual_score(individual, self.estimators, self.X, self.y, self.classes) - initial_score) * 100) for individual in top_individuals])
		print([hash(e) for e in [''.join([str(s) for s in x]) for x in top_individuals]])

		return best_score, scores


	def run_genetic_evolution(self):		
		initial_score = majority_voting_score(self.X, self.y, self.estimators, self.classes)
		print("\nInitial score = %f" % initial_score)
		prev_score = 0

		print("\nGeneration: 0")				
		scores = self.__parallel_score_processing()

		for i in range(self.iterations):
			
			print("Calculating population fitness...")	
			fitness_prob = self.__calculate_fitness_probabilities(scores)
			
			print("Reproducing population...        ")
			self.__update_pop_history()
			self.pop = self.__reproduce_population(fitness_prob)

			print("Calculating statistics...   ")
			prev_score, scores = self.__calculate_population_stats(initial_score, prev_score)

			print("\nGeneration: %d" % (i + 1))
