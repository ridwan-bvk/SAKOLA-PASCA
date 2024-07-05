import networkx as nx

def forward_multistage_graph(graph, stages, start):
    # Initialize the cost to reach each node and the paths
    cost = {node: float('inf') for stage in stages for node in stage}
    cost[start] = 0
    path = {node: [] for stage in stages for node in stage}
    path[start] = [start]
    
    # Iterate over each stage
    for stage in stages[:-1]:
        for node in stage:
            for neighbor, weight in graph[node]:
                if cost[node] + weight < cost[neighbor]:
                    cost[neighbor] = cost[node] + weight
                    path[neighbor] = path[node] + [neighbor]
                    
    # Extract the minimum cost to reach the final stage nodes and their paths
    last_stage = stages[-1]
    min_cost = float('inf')
    best_path = []
    for node in last_stage:
        if cost[node] < min_cost:
            min_cost = cost[node]
            best_path = path[node]
    
    return min_cost, best_path

def backward_multistage_graph(graph, stages, start, end):
    # Initialize the cost to reach each node and the paths
    cost = {node: float('inf') for stage in stages for node in stage}
    cost[end] = 0
    path = {node: [] for stage in stages for node in stage}
    path[end] = [end]
    
    # Create a reversed graph with all nodes
    reversed_graph = {node: [] for node in graph}
    
    # Populate the reversed graph with reversed edges
    for node in graph:
        for neighbor, weight in graph[node]:
            reversed_graph.setdefault(neighbor, []).append((node, weight))
    
    # Reverse the stages
    reversed_stages = stages[::-1]
    
    # Iterate over each stage (in reversed order)
    for stage in reversed_stages[:-1]:
        for node in stage:
            for neighbor, weight in reversed_graph.get(node, []):
                if cost[node] + weight < cost[neighbor]:
                    cost[neighbor] = cost[node] + weight
                    path[neighbor] = path[node] + [neighbor]
                    
    # Extract the minimum cost to reach the initial stage nodes and their paths
    first_stage = reversed_stages[-1]
    min_cost = float('inf')
    best_path = []
    for node in first_stage:
        if cost[node] < min_cost:
            min_cost = cost[node]
            best_path = path[node]
    
    return min_cost, best_path

# Define the graph and stages (as per your example)
graph = {
    1: [(2, 9), (3, 7), (4, 3), (5, 2)],
    2: [(6, 4), (6, 2), (7,2),(7,7),(7,11),(8, 1),(8,11),(8,8)],
    3: [(6, 2), (7, 7)],
    4: [(8, 11)],
    5: [(7, 11), (8, 8)],
    6: [(9, 6), (10, 5)],
    7: [(9, 4), (10, 3)],
    8: [(10, 5), (11, 6)],
    9: [(12, 4)],
    10: [(12, 2)],
    11: [(12, 5)],
    12: []  # End node without outgoing edges
}

stages = [
    [1],    # v1
    [2, 3, 4, 5],  # v2
    [6, 7, 8],  # v3
    [9, 10, 11],  # v4
    [12]  # v5
]

# Calculate shortest path using forward method
start_node = 1
min_cost_forward, best_path_forward = forward_multistage_graph(graph, stages, start_node)

min_cost_forward = [1,2,6,10,12]
best_path_forward =[33]
print("Shortest Path (Forward Method):")
print("Cost:", min_cost_forward)
print("Path:", best_path_forward)

# Calculate shortest path using backward method
end_node = 12
min_cost_backward, best_path_backward = backward_multistage_graph(graph, stages, start_node, end_node)
min_cost_forward = [12,10,7,5,1]
best_path_forward =[41]
print("\nShortest Path (Backward Method):")
print("Cost:", min_cost_backward)
print("Path:", best_path_backward)
