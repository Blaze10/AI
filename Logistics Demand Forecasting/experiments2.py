from DemandPrediction import DemandPrediction
from GeneticProgramming import GeneticProgramming
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np


# Create an instance of the demand prediction problem for the training dataset
demand_train = DemandPrediction("train")

# Create an instance of the genetic programming algorithm for the demand prediction problem
gp = GeneticProgramming(demand_train)

# Set up the hyperparameters to tune
population_sizes = [50, 100, 200]
generations = [50, 100, 200]
mutation_rates = [0.05, 0.1, 0.2]
tournament_sizes = [3, 5, 7]
experiment_number = 0
experiment_results = {}
mae_train_list = []
mae_test_list = []


# Perform a grid search over the hyperparameters and evaluate the MAE on the training and test datasets
for p in population_sizes:
    for g in generations:
        for m in mutation_rates:
            for t in tournament_sizes:
                gp.POPULATION_SIZE = p
                gp.GENERATIONS = g
                gp.MUTATION_RATE = m
                gp.TOURNAMENT_SIZE = t
                population = gp.generate_population()
                best_individual = None
                for i in range(gp.GENERATIONS):
                    new_population = []
                    for j in range(gp.POPULATION_SIZE):
                        parent1 = gp.tournament_selection(population)
                        parent2 = gp.tournament_selection(population)
                        child = gp.crossover(parent1, parent2)
                        mutated_child = gp.mutate(child)
                        new_population.append(mutated_child)
                    population = new_population
                    best_individual = min(population, key=gp.evaluate_fitness)
                demand_test = DemandPrediction("test")
                mae_train = gp.evaluate_fitness(best_individual)
                mae_test = demand_test.evaluate(best_individual)
                experiment_number += 1
                experiment_results[experiment_number] = {
                    'Population size': p,
                    'Generations': g,
                    'Mutation rate': m,
                    'Tournament size': t
                }

                print('########################################################################################')
                print(f'Experiment number: {experiment_number}')
                print(f"Population size: {p}, Generations: {g}, Mutation rate: {m}, Tournament size: {t}")
                print(f"MAE on training dataset: {mae_train}")
                print(f"MAE on test dataset: {mae_test}")
                print('########################################################################################')

df = pd.DataFrame.from_dict(experiment_results)
df

# Tests
# Shapiro-Wilk normality test and Levene's test for equal variances on the lists of MAE values for each hyperparameter configuration:
sw_train, p_train = stats.shapiro(mae_train_list)
sw_test, p_test = stats.shapiro(mae_test_list)
levene, p_levene = stats.levene(mae_train_list, mae_test_list)

print(f"Shapiro-Wilk normality test on MAE values for training dataset: p-value = {p_train:.4f}")
print(f"Shapiro-Wilk normality test on MAE values for test dataset: p-value = {p_test:.4f}")
print(f"Levene's test for equal variances of MAE values: p-value = {p_levene:.4f}")


# Plot
plt.scatter(range(len(mae_train_list)), mae_train_list, label='train')
plt.scatter(range(len(mae_test_list)), mae_test_list, label='test')
plt.xlabel('Configuration')
plt.ylabel('MAE')
plt.legend()
plt.show()

f, p_anova = stats.f_oneway(mae_train_list, mae_test_list)
print(f"ANOVA test for significant differences in MAE values: p-value = {p_anova:.4f}")




