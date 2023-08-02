from helper import getCostOfRoute, readCsvFile
from bat_algorithm import bat_algorithm
from flower_pollination import flower_pollination_algorithm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def main():
    # Read the TSP data from the CSV file
    filename = 'data.csv'
    graph = readCsvFile(filename)

    # Parameters for the Bat Algorithm
    population_size = 50
    max_iter = 1000
    alpha = 0.1

    # Run the Bat Algorithm
    best_bat, bat_cost_history = bat_algorithm(graph, population_size, max_iter, alpha)
    print("Best route found by the Bat Algorithm: ", best_bat)
    print("Cost of the best route found by the Bat Algorithm: ", getCostOfRoute(best_bat, graph))

    # Parameters for the Flower Pollination Algorithm
    p = 0.8

    # Run the Flower Pollination Algorithm
    best_flower, flower_cost_history = flower_pollination_algorithm(graph, population_size, max_iter, p)
    print("Best route found by the Flower Pollination Algorithm: ", best_flower)
    print("Cost of the best route found by the Flower Pollination Algorithm: ", getCostOfRoute(best_flower, graph))
    # animate_convergence(bat_cost_history, flower_cost_history, 'convergence_animation.gif')

    # Visualize the best routes found by both algorithms
    plt.figure()
    plt.plot(bat_cost_history, label='Bat Algorithm')
    plt.plot(flower_cost_history, label='Flower Pollination Algorithm')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()



def animate_convergence(bat_cost_history, flower_cost_history, filename):
    fig, ax = plt.subplots()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title('Convergence of Bat Algorithm and Flower Pollination Algorithm')

    bat_line, = ax.plot([], [], label='Bat Algorithm', lw=2)
    flower_line, = ax.plot([], [], label='Flower Pollination Algorithm', lw=2)

    ax.legend()

    def update(frame):
        ax.clear()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.set_title('Convergence of Bat Algorithm and Flower Pollination Algorithm')
        ax.plot(bat_cost_history[:frame+1], label='Bat Algorithm', lw=2)
        ax.plot(flower_cost_history[:frame+1], label='Flower Pollination Algorithm', lw=2)
        ax.legend()

    frames = []
    for frame in range(len(bat_cost_history)):
        update(frame)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(Image.fromarray(image))

    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=100, loop=0)
    plt.show()


main()