import random
from operator import add, sub, mul
from DemandPrediction import DemandPrediction
import time


class GeneticProgramming:
    POPULATION_SIZE = 100
    GENERATIONS = 100
    MUTATION_RATE = 0.1
    TOURNAMENT_SIZE = 5

    def __init__(self, problem):
        self.problem = problem

    # Generate an initial population of random individuals
    def generate_population(self):
        population = []
        for i in range(GeneticProgramming.POPULATION_SIZE):
            individual = []
            for j in range(DemandPrediction.N_PARAMETERS):
                individual.append(random.uniform(-100, 100))
            population.append(individual)
        return population

    # Evaluate the fitness of an individual by calculating its mean absolute
    # error on the training dataset
    def evaluate_fitness(self, individual):
        return self.problem.evaluate(individual)

    # Select individuals for the next generation using tournament selection
    def tournament_selection(self, population):
        tournament = random.sample(population, GeneticProgramming.TOURNAMENT_SIZE)
        best_individual = min(tournament, key=self.evaluate_fitness)
        return best_individual

    # Crossover two individuals to create a new offspring
    def crossover(self, parent1, parent2):
        child = []
        for i in range(DemandPrediction.N_PARAMETERS):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    # Mutate an individual by replacing one of its parameters with a random value
    def mutate(self, individual):
        mutated_individual = individual.copy()
        if random.random() < GeneticProgramming.MUTATION_RATE:
            index = random.randint(0, DemandPrediction.N_PARAMETERS - 1)
            mutated_individual[index] = random.uniform(-100, 100)
        return mutated_individual

    # Run the genetic programming algorithm
    def run(self):
        start_time = time.time()
        population = self.generate_population()
        best_fitness_scores = []
        best_individual = None
        for i in range(GeneticProgramming.GENERATIONS):
            print("Generation ", i + 1)
            new_population = []
            new_population = []
            fitness_scores = []
            for j in range(GeneticProgramming.POPULATION_SIZE):
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
                fitness_scores.append(self.evaluate_fitness(child))
            population = new_population
            best_fitness_score = min(fitness_scores)
            best_fitness_scores.append(best_fitness_score)
            if best_individual is None or self.evaluate_fitness(best_individual) > best_fitness_score:
                best_individual = population[fitness_scores.index(best_fitness_score)]

        best_individual = min(population, key=self.evaluate_fitness)

        end_time = time.time()
        time_taken = end_time - start_time
        print("Best individual: ", best_individual)
        print("Fitness: ", self.evaluate_fitness(best_individual))
        print("Best fitness score: ", min(best_fitness_scores))
        print("Best individual: ", best_individual)
        print("Time taken: ", time_taken)


problem = DemandPrediction("train")
gp = GeneticProgramming(problem)
gp.run()
