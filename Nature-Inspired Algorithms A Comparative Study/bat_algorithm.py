import random
import numpy as np
from helper import getCostOfRoute

def initialize_bat_population(graph, population_size):
    bats = []
    for _ in range(population_size):
        bat = list(range(len(graph)))
        random.shuffle(bat)
        bats.append(bat)
    return bats

def update_velocity(bats, velocities, best_bat, graph, alpha):
    for i in range(len(bats)):
        velocities[i] = (np.array(velocities[i]) + alpha * (np.array(best_bat) - np.array(bats[i]))).tolist()

def update_position(bats, velocities):
    for i in range(len(bats)):
        for j in range(len(bats[i])):
            r = random.random()
            if r < abs(velocities[i][j]):
                swap_idx = int((j + abs(velocities[i][j])) % len(bats[i]))
                bats[i][j], bats[i][swap_idx] = bats[i][swap_idx], bats[i][j]

def local_search(bat, graph):
    i, j = random.sample(range(len(bat)), 2)
    new_bat = bat.copy()
    new_bat[i], new_bat[j] = new_bat[j], new_bat[i]
    if getCostOfRoute(new_bat, graph) < getCostOfRoute(bat, graph):
        return new_bat
    return bat

def bat_algorithm(graph, population_size, max_iter, alpha):
    bats = initialize_bat_population(graph, population_size)
    velocities = [[0 for _ in range(len(graph))] for _ in range(population_size)]
    best_bat = min(bats, key=lambda x: getCostOfRoute(x, graph))

    # Initialize the cost history
    cost_history = []

    for _ in range(max_iter):
        update_velocity(bats, velocities, best_bat, graph, alpha)
        update_position(bats, velocities)

        for i in range(len(bats)):
            bats[i] = local_search(bats[i], graph)

        current_best_bat = min(bats, key=lambda x: getCostOfRoute(x, graph))
        current_best_cost = getCostOfRoute(current_best_bat, graph)
        cost_history.append(current_best_cost)

        if current_best_cost < getCostOfRoute(best_bat, graph):
            best_bat = current_best_bat

    return best_bat, cost_history