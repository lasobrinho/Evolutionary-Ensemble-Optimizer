
import numpy as np
from random import randint
from colorama import init, Fore, Style
from metrics import majority_voting_score
from sklearn.externals.joblib import Parallel, delayed

from collections import Counter


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
                 iterations=1000, 
                 n_jobs=1):     
        self.estimators = estimators
        self.classes = classes
        self.X = data
        self.y = target
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.iterations = iterations
        self.n_jobs = n_jobs
        self.no_score_change = 0
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

    def __generate_child(self, pair):
        cut_index = randint(1, len(pair[0][1]) - 2)
        individual_1 = pair[0]
        individual_2 = pair[1]
        child = individual_1[1][:cut_index] + individual_2[1][cut_index:]
        return child

    def __crossover(self, pair, crossover_pop=None):
        child = self.__generate_child(pair)
        if crossover_pop:
            original_child = child
            while child in crossover_pop:
                child = self.__soft_mutate(original_child[:])
            if original_child != child:
                self.soft_mutations += 1
        return child

    def __soft_mutate(self, individual):
        index_mutation = randint(0, len(individual) - 1)
        individual[index_mutation] ^= 1
        return individual

    def __mutate(self, individual):
        if np.random.rand() <= self.mutation_rate:
            self.natural_mutations += 1
            n_mutations = randint(1, len(individual) // 8)
            for _ in range(n_mutations):
                index_mutation = randint(0, len(individual) - 1)
                individual[index_mutation] ^= 1
        return individual

    def __reproduce_population(self, fitness_prob, sel_sensivity=None):
        sorted_pop = [pop for _, pop in sorted(zip(fitness_prob, self.pop))]
        sorted_pop = [(idx, ind) for idx, ind in zip(list(range(len(sorted_pop))), sorted_pop)]

        if sel_sensivity:
            a = np.arange(1, len(sorted_pop) + 1)
            sel_prob = [((sel_sensivity - 1) / ((sel_sensivity**len(a)) - (1))) * (sel_sensivity**(len(a)-i)) for i in a]
            n_promoted = 0
        else:
            sel_prob = sorted(fitness_prob)
            n_promoted = 2

        new_pop = []
        crossover_pop = []
        for i in range(len(self.pop) - n_promoted):
            pair_indexes = np.random.choice(len(sorted_pop), 2, replace=False, p=sel_prob).tolist()
            pair = [sorted_pop[pair_indexes[0]], sorted_pop[pair_indexes[1]]]
            child = self.__crossover(pair, crossover_pop)
            crossover_pop.append(child)
        
        new_pop = [self.__mutate(individual) for individual in crossover_pop]
        if not sel_sensivity:
            new_pop += [e[1] for e in sorted_pop[-n_promoted:]]

        return new_pop

    def __get_population_diversity(self):
        duplicates = dict(Counter([''.join([str(s) for s in i]) for i in self.pop]))
        duplicates_count = np.sum([duplicates[k] - 1 for k in duplicates.keys()])
        return ((len(self.pop) - duplicates_count) / len(self.pop)), duplicates_count

    def __rank_population(self):
        scores = self.__parallel_score_processing()
        sorted_pop = [individual for _, individual in sorted(zip(scores, self.pop), reverse=True)]
        return sorted_pop, scores

    def __calculate_population_stats(self, initial_score, prev_score):
        sorted_pop, scores = self.__rank_population()
        best_individual = sorted_pop[0]
        best_score = _get_individual_score(best_individual, self.estimators, self.X, self.y, self.classes)
        score_diff = (best_score - initial_score) * 100
        if best_score > prev_score:
            self.no_score_change = 0
            print("Best score:      " + Fore.GREEN + Style.BRIGHT + "%f%%" % (best_score * 100) + Style.RESET_ALL + " (%f%%)" % (score_diff) + " (%d) " % self.no_score_change)
        elif best_score < prev_score:
            print("Best score:      " + Fore.RED + Style.BRIGHT + "%f%%" % (best_score * 100) + Style.RESET_ALL + " (%f%%)" % (score_diff) + " (%d) " % self.no_score_change)
        else:
            self.no_score_change += 1
            print("Best score:      " + Fore.CYAN + Style.BRIGHT + "%f%%" % (best_score * 100) + Style.RESET_ALL + " (%f%%)" % (score_diff) + " (%d) " % self.no_score_change)

        print("Average score:   %f%%" % (np.mean(scores) * 100))
        print("Standard dev.:    %f%%" % (np.std(scores) * 100))
        pop_diversity, pop_duplicates = self.__get_population_diversity()
        print("Pop. diversity: %.2f%%" % (pop_diversity * 100) + " (%d duplicates)" % pop_duplicates)

        return best_score, scores, best_individual


    def run_genetic_evolution(self):        
        initial_score = majority_voting_score(self.X, self.y, self.estimators, self.classes)
        print("\nInitial score = %f%%" % (initial_score * 100))
        prev_score = 0

        print("\n_________________________________________________________________")
        print("Generation: 0")                
        scores = self.__parallel_score_processing()

        for i in range(self.iterations):
            
            fitness_prob = self.__calculate_fitness_probabilities(scores)
            
            print("\nReproducing population...        ")
            self.natural_mutations = 0
            self.soft_mutations = 0
            self.pop = self.__reproduce_population(fitness_prob, sel_sensivity=0.85)
            print("Natural mutations: %d" % (self.natural_mutations) + " (%.2f%%)" % ((self.natural_mutations / len(self.pop)) * 100))
            print("Soft mutations:    %d" % (self.soft_mutations) + " (%.2f%%)" % ((self.soft_mutations / len(self.pop)) * 100))

            print("\nCalculating statistics...   ")
            prev_score, scores, best_individual = self.__calculate_population_stats(initial_score, prev_score)

            print("\n_________________________________________________________________")
            print("Generation: %d" % (i + 1))

        return best_individual, prev_score
