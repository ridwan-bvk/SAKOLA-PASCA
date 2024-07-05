import networkx as nx
import matplotlib.pyplot as plt

# Definisi graf menggunakan dictionary
graph = {
    1: [2, 3, 4],
    2: [5, 6],
    5: [9, 10],
    4: [7, 8],
    7: [11, 12],
    # Nodes with no outgoing edges are omitted since they won't affect traversal.
}

# Membuat objek graf
G = nx.DiGraph(graph)

# Mengatur posisi node secara manual untuk visualisasi yang lebih rapi
pos = {
    1: (0, 0),
    2: (-1, -1),
    3: (0, -1),
    4: (1, -1),
    5: (-2, -2),
    6: (-1, -2),
    7: (1, -2),
    8: (2, -2),
    9: (-3, -3),
    10: (-2, -3),
    11: (1, -3),
    12: (2, -3),
}

# Inisialisasi level order traversal menggunakan BFS
def level_order_traversal(start):
    if start not in graph:
        return []
    
    queue = [start]
    result = []
    visited = set(queue)
    
    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
    
    return result

# Pre order traversal
def pre_order_traversal(node):
    if node is None:
        return []
    
    result = []
    stack = [node]
    
    while stack:
        current = stack.pop()
        result.append(current)
        children = graph.get(current, [])
        stack.extend(reversed(children))  # Reverse to maintain left-to-right order
    
    return result

# In order traversal
def in_order_traversal(node):
    if node is None:
        return []
    
    result = []
    stack = []
    current = node
    
    while stack or current:
        while current:
            stack.append(current)
            current = graph.get(current, [])[0] if graph.get(current) else None  # Go leftmost
        
        current = stack.pop()
        result.append(current)
        current = graph.get(current, [])[1] if graph.get(current) and len(graph.get(current)) > 1 else None  # Go right
        
    return result

# Post order traversal
def post_order_traversal(node):
    if node is None:
        return []
    
    result = []
    stack = [node]
    
    while stack:
        current = stack.pop()
        result.append(current)
        children = graph.get(current, [])
        stack.extend(children)  # Extend with children to process them next
    
    return result[::-1]  # Reverse the result to get post-order

# Calculate and print traversals
print("Level Order Traversal:", level_order_traversal(1))
print("Pre Order Traversal:", pre_order_traversal(1))
print("In Order Traversal:", in_order_traversal(1))
print("Post Order Traversal:", post_order_traversal(1))

# Draw the graph for visualization
plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=12, font_weight='bold', arrows=True)
plt.title('Traversal Graph')
plt.show()
