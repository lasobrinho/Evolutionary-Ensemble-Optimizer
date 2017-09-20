
import numpy as np
from random import randint
from colorama import init, Fore, Style
from metrics import majority_voting_score
from sklearn.externals.joblib import Parallel, delayed

from collections import Counter


def get_individual_score(individual, estimators, X, y, classes):
    selected_estimators = [estimator for estimator, isSelected in zip(estimators, individual) if isSelected]
    individual_score = majority_voting_score(X, y, selected_estimators, classes)
    return individual_score

def _get_population_scores(pop_slice, estimators, X, y, classes):
    return [get_individual_score(individual, estimators, X, y, classes) for individual in pop_slice]

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
                 elitism=False,
                 n_point_crossover=False,
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
        self.elitism = elitism
        self.n_point_crossover = n_point_crossover
        self.no_score_change = 0
        self.duplicates_count = 0
        self.pop_history = []
        self.best_so_far = (0.0, [])
        self.__generate_random_population()
        init()

    def __generate_random_population(self):
        print("\nGenerating random population")
        print("Population size = %d" % self.pop_size)
        print("Individual size (genes) = %d" % len(self.estimators))
        self.pop = []
        for _ in range(self.pop_size):
            individual = []
            for _ in range(len(self.estimators)):
                individual.append(randint(0, 1))
            self.pop.append(individual)
            self.__update_pop_history(individual)

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
        scores_mean = np.mean(scores)
        scores_std = np.std(scores)

        new_scores = [((score + (score - scores_mean) * 1000)**((score - scores_mean)/scores_std)) if ((score + (score - scores_mean) * 1000)**((score - scores_mean)/scores_std)) > 0 else 0.1 for score in scores]

        scores_sum = np.sum(new_scores)
        reproduction_prob = [score/scores_sum for score in new_scores]
        return reproduction_prob

    def __generate_child(self, pair, n_point_crossover):
        parent_1 = pair[0]
        parent_2 = pair[1]
        if n_point_crossover:
            n_cuts = randint(1, len(self.estimators) // 2)
            cut_indexes = sorted([randint(1, len(parent_1[1])-2) for i in list(range(n_cuts))] + [0, len(parent_1[1])])
            child_lists = [parent_1[1][cut_indexes[cut_indexes.index(i)-1]:i] if (cut_indexes[1:].index(i) % 2 == 0) else parent_2[1][cut_indexes[cut_indexes.index(i)-1]:i] for i in cut_indexes[1:]]
            child = [e for sublist in child_lists for e in sublist]
        else:
            cut_index = randint(1, len(pair[0][1]) - 2)
            child = parent_1[1][:cut_index] + parent_2[1][cut_index:]
        return child

    def __crossover(self, pair, n_point_crossover, crossover_pop=None):
        if np.random.rand() <= self.crossover_rate:
            child = self.__generate_child(pair, n_point_crossover)
            original_child = child[:]
            while child in self.pop_history:
                self.children_rejected += 1
                child = self.__soft_mutate(child)
            if original_child != child:
                self.forced_mutations += 1
        else:
            self.skipped_crossover += 1
            r = np.random.rand()
            if r < 0.5:
                child = pair[0][1]
            else:
                child = pair[1][1]
        self.__update_pop_history(child)
        return child

    def __soft_mutate(self, individual):
        index_mutation = randint(0, len(individual) - 1)
        individual[index_mutation] ^= 1
        return individual

    def __mutate(self, individual):
        if np.random.rand() <= self.mutation_rate:
            self.natural_mutations += 1
            n_mutations = randint(1, len(individual) // 8)
            index_history = []
            for _ in range(n_mutations):
                index_mutation = randint(0, len(individual) - 1)
                while index_mutation in index_history:
                    index_mutation = randint(0, len(individual) - 1)
                index_history.append(index_mutation)
                individual[index_mutation] ^= 1
        return individual

    def __random_selection(self, sorted_pop, sel_prob):
        pair_indexes = np.random.choice(len(sorted_pop), 2, replace=False, p=sel_prob).tolist()
        pair = [sorted_pop[pair_indexes[0]], sorted_pop[pair_indexes[1]]]
        return pair

    def __adjust_sel_sensivity(self, sel_sensivity):
        return min(sel_sensivity + (self.no_score_change * 0.010), 0.950)

    def __reproduce_population(self, fitness_prob, n_point_crossover=False, sel_sensivity=None, elitism=False):
        sorted_pop = [pop for _, pop in sorted(zip(fitness_prob, self.pop))]
        sorted_pop = [(idx, ind) for idx, ind in zip(list(range(len(sorted_pop))), sorted_pop)]

        if sel_sensivity:
            sel_sensivity = self.__adjust_sel_sensivity(sel_sensivity)
            print("** sel_sensivity = %f" % sel_sensivity)
            a = np.arange(1, len(sorted_pop) + 1)
            sel_prob = [((sel_sensivity - 1) / ((sel_sensivity**len(a)) - (1))) * (sel_sensivity**(len(a)-i)) for i in a]
        else:
            sel_prob = sorted(fitness_prob)

        if elitism:
            n_promoted = 1
            elite = [e[1][:] for e in sorted_pop[-n_promoted:]]
        else:
            n_promoted = 0

        new_pop = []
        crossover_pop = []

        if elitism:
            for e in elite:
                if e not in self.pop_history:
                    self.__update_pop_history(e)

        for i in range(len(self.pop) - n_promoted):
            pair = self.__random_selection(sorted_pop, sel_prob)
            child = self.__crossover(pair, n_point_crossover, crossover_pop)
            crossover_pop.append(child)
        
        new_pop = [self.__mutate(individual) for individual in crossover_pop]
        if elitism:
            new_pop += elite
        return new_pop

    def __update_duplicates(self):
        duplicates = dict(Counter([''.join([str(s) for s in i]) for i in self.pop]))
        self.duplicates_count = np.sum([duplicates[k] - 1 for k in duplicates.keys()])

    def __get_population_diversity(self):
        return ((len(self.pop) - self.duplicates_count) / len(self.pop))

    def __rank_population(self):
        scores = self.__parallel_score_processing()
        sorted_pop = [individual for _, individual in sorted(zip(scores, self.pop), reverse=True)]
        return sorted_pop, scores

    def __remove_outperformers(self, scores):
        mean_score = np.mean(scores)
        i = 0
        new_pop = []
        for individual in self.pop:
            if scores[i] < mean_score:
                new_individual = []
                for _ in range(len(self.estimators)):
                    new_individual.append(randint(0, 1))
                self.pop[i] = new_individual
            i += 1


    def __calculate_population_stats(self, initial_score, prev_score):
        sorted_pop, scores = self.__rank_population()
        best_individual = sorted_pop[0]
        best_score = get_individual_score(best_individual, self.estimators, self.X, self.y, self.classes)
        if best_score > self.best_so_far[0]:
            self.best_so_far = (best_score, best_individual)
        score_diff = (best_score - initial_score) * 100
        if best_score > prev_score:
            self.no_score_change = 0
            print("Best in population: " + Fore.GREEN + Style.BRIGHT + "%f%%" % (best_score * 100) + Style.RESET_ALL + " (%f%%)" % (score_diff) + " (%d) " % self.no_score_change)
        elif best_score < prev_score:
            print("Best in population: " + Fore.RED + Style.BRIGHT + "%f%%" % (best_score * 100) + Style.RESET_ALL + " (%f%%)" % (score_diff) + " (%d) " % self.no_score_change)
        else:
            self.no_score_change += 1
            print("Best in population: " + Fore.CYAN + Style.BRIGHT + "%f%%" % (best_score * 100) + Style.RESET_ALL + " (%f%%)" % (score_diff) + " (%d) " % self.no_score_change)

        print("Average score:      %f%%" % (np.mean(scores) * 100))
        print("Standard dev.:       %f%%" % (np.std(scores) * 100))
        pop_diversity = self.__get_population_diversity()
        print("Chromossomes:       %d" % len(self.pop_history))
        print("Pop. diversity:     %.2f%%" % (pop_diversity * 100) + " (%d duplicates)" % self.duplicates_count)
        print("Best score so far:  %f%%" % (self.best_so_far[0] * 100) + " (%f%%)" % ((self.best_so_far[0] - initial_score)*100))

        # self.__remove_outperformers(scores)

        return best_score, scores, best_individual

    def __update_pop_history(self, individual):
        self.pop_history += [individual]

    def run_genetic_evolution(self):
        # self.weights = get_weights(self.X, self.y, self.estimators, self.classes)
        initial_score = majority_voting_score(self.X, self.y, self.estimators, self.classes)        
        print("\nInitial score = %f%%" % (initial_score * 100))
        self.best_so_far = (initial_score, [])
        # input()
        prev_score = 0

        print("\n________________________________________________________________________________")
        print("Generation: 0")
        scores = self.__parallel_score_processing()

        for i in range(self.iterations):

            try:
                fitness_prob = self.__calculate_fitness_probabilities(scores)
        
                print("\nReproducing population...        ")
                self.natural_mutations = 0
                self.forced_mutations = 0
                self.skipped_crossover = 0
                self.children_rejected = 0
                # self.pop = self.__reproduce_population(fitness_prob, sel_sensivity=0.95, elitism=True)
                self.pop = self.__reproduce_population(fitness_prob, n_point_crossover=self.n_point_crossover, elitism=self.elitism)
                self.__update_duplicates()
                print("Natural mutations: %d" % (self.natural_mutations) + " (%.2f%%)" % ((self.natural_mutations / len(self.pop)) * 100))
                print("Forced mutations:  %d" % (self.forced_mutations) + " (%.2f%%)" % ((self.forced_mutations / len(self.pop)) * 100))
                print("Skipped crossover: %d" % (self.skipped_crossover) + " (%.2f%%)" % ((self.skipped_crossover / len(self.pop)) * 100))
                print("Children rejected: %d" % (self.children_rejected))

                print("\nCalculating statistics...   ")
                prev_score, scores, best_individual = self.__calculate_population_stats(initial_score, prev_score)

                print("\n________________________________________________________________________________")
                print("Generation: %d" % (i + 1))
            except KeyboardInterrupt:
                break

        print("\n\n\n================================================================================")
        print("\nFinished genetic optimization")
        return self.best_so_far, initial_score
