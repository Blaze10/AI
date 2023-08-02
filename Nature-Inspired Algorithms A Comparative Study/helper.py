import math
import csv

def readCsvFile(filename):
    # initalize the graph as empty list
    graph = []

    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        next(reader)  # skip first row
        data = [row for row in reader]

    # calculate the distance between each pair of cities and populate the graph
    for i in range(len(data)):
        row = []
        for j in range(len(data)):
            x1, y1 = float(data[i][1]), float(data[i][2])
            x2, y2 = float(data[j][1]), float(data[j][2])
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            row.append(distance)
        graph.append(row)
    return graph

def getCostOfRoute(route, graph):
    cost = 0
    for i in range(len(route)-1):
        cost += graph[route[i]][route[i+1]]
    return cost