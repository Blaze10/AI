import random
from operator import add, sub, mul
from DemandPrediction import DemandPrediction


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
        population = self.generate_population()
        for i in range(GeneticProgramming.GENERATIONS):
            print("Generation ", i + 1)
            new_population = []
            for j in range(GeneticProgramming.POPULATION_SIZE):
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            population = new_population
        best_individual = min(population, key=self.evaluate_fitness)
        print("Best individual: ", best_individual)
        print("Fitness: ", self.evaluate_fitness(best_individual))


problem = DemandPrediction("train")
gp = GeneticProgramming(problem)
gp.run()