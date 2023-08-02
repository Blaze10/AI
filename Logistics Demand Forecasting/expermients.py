from DemandPrediction import DemandPrediction
from GeneticProgramming import GeneticProgramming
import pandas as pd

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