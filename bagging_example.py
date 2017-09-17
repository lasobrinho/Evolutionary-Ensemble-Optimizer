
import numpy as np
import string
from optimizers import GeneticOptimizer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


if __name__ == "__main__":

    print("\n---------------------------------------")

    datasetFolderName = 'UCI_Datasets/'
    datasetFileName = 'letter-recognition.data'

    letter_mapping = list(string.ascii_uppercase)
    letter_dataset = np.loadtxt(datasetFolderName + datasetFileName, delimiter=",")
    letter_data = letter_dataset[:, 1:16]
    letter_target = letter_dataset[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(letter_data, letter_target, test_size=0.5)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.25)


    n_estimators = 100
    pop_size = n_estimators // 2
    iterations = 5000   
    mutation_rate = 0.25
    crossover_rate = 1.0
    n_jobs = 8

    print("\nGenerating estimators from Bagging method")
    max_samples_ratio = 0.5
    bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), bootstrap=False, n_estimators=n_estimators, max_samples=max_samples_ratio)
    bagging.fit(X_train, y_train)


    gen_opt = GeneticOptimizer(estimators=bagging.estimators_, 
                               classes=bagging.classes_, 
                               data=X_test, 
                               target=y_test, 
                               pop_size=pop_size, 
                               mutation_rate=mutation_rate, 
                               crossover_rate=crossover_rate, 
                               iterations=iterations, 
                               n_jobs=n_jobs)

    best_subset_ensemble, best_score = gen_opt.run_genetic_evolution()

    print(best_score)
    print(best_subset_ensemble)
