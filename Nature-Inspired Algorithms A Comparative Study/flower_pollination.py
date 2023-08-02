import random
from helper import getCostOfRoute

def initialize_flower_population(graph, population_size):
    flowers = []
    for _ in range(population_size):
        flower = list(range(len(graph)))
        random.shuffle(flower)
        flowers.append(flower)
    return flowers

def flower_pollination_algorithm(graph, population_size, max_iter, p):
    flowers = initialize_flower_population(graph, population_size)
    best_flower = min(flowers, key=lambda x: getCostOfRoute(x, graph))

    # Initialize the cost history
    cost_history = []

    for _ in range(max_iter):
        flowers = global_pollination(graph, flowers, best_flower, p)

        current_best_flower = min(flowers, key=lambda x: getCostOfRoute(x, graph))
        current_best_cost = getCostOfRoute(current_best_flower, graph)
        cost_history.append(current_best_cost)

        if current_best_cost < getCostOfRoute(best_flower, graph):
            best_flower = current_best_flower

    return best_flower, cost_history

def global_pollination(graph, flowers, best_flower, p):
    new_flowers = []
    for flower in flowers:
        if random.random() < p:
            new_flower = flower.copy()
            i, j = random.sample(range(len(flower)), 2)
            new_flower[i], new_flower[j] = new_flower[j], new_flower[i]  # Swap elements in the current flower
            if getCostOfRoute(new_flower, graph) < getCostOfRoute(flower, graph):
                new_flowers.append(new_flower)
            else:
                new_flowers.append(flower)
        else:
            new_flowers.append(flower)
    return new_flowers