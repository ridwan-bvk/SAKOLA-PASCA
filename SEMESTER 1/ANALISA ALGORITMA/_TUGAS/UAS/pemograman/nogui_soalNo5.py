import networkx as nx
import matplotlib.pyplot as plt
import math

# Definisikan fungsi DFS untuk mencari jalur terpendek
def dfs(graph, start, path, visited, total_distance):
    global shortest_path, shortest_distance
    
    # Tandai node saat ini sebagai sudah dikunjungi dan tambahkan ke jalur
    visited.add(start)
    path.append(start)
    
    # Jika semua node sudah dikunjungi, periksa apakah jalur saat ini adalah yang terpendek
    if len(visited) == len(graph):
        if total_distance < shortest_distance:
            shortest_distance = total_distance
            shortest_path = path.copy()
    else:
        # Jelajahi setiap tetangga
        for neighbor, distance in graph[start].items():
            if neighbor not in visited:
                dfs(graph, neighbor, path, visited, total_distance + distance)
    
    # Backtrack
    visited.remove(start)
    path.pop()

# Fungsi untuk menggambar grafik dengan highlight jalur terpendek
def plot_graph_with_shortest_path(graph, positions, shortest_path):
    G = nx.Graph()
    G.add_nodes_from(graph.keys())
    
    # Tambahkan edge dengan bobot
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)
    
    # Atur posisi node
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos=positions, with_labels=True, node_color='lightblue', node_size=500, font_size=12, font_weight='bold')
    
    # Highlight jalur terpendek
    edges = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)]
    edge_colors = ['red' if edge in edges else 'black' for edge in G.edges()]
    nx.draw_networkx_edges(G, pos=positions, edgelist=G.edges(), edge_color=edge_colors, width=2.0)
    
    # Tampilkan bobot pada edge
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=labels, font_color='blue')
    
    # Tampilkan grafik
    plt.title('Shortest Path Visualization')
    plt.show()

# Inisialisasi variabel global untuk menyimpan jalur terpendek dan jaraknya
shortest_path = []
shortest_distance = float('inf')

# Definisikan grafik dengan bobot
graph = {
    'P': {'R': 10, 'S': 9, 'Q': 20},
    'R': {'P': 10, 'S': 17, 'Q': 13},
    'S': {'P': 9, 'R': 17, 'Q': 15},
    'Q': {'P': 20, 'R': 13, 'S': 15}
}

# Definisikan posisi node untuk plotting
positions = {
    'P': (100, 50),
    'R': (300, 50),
    'S': (100, 300),
    'Q': (300, 300)
}

# Cari jalur terpendek menggunakan DFS
dfs(graph, 'P', [], set(), 0)

# Tampilkan jalur terpendek dalam teks
print(f"Shortest path: {' -> '.join(shortest_path)}")
print(f"Shortest distance: {shortest_distance}")

# Plot grafik dengan highlight jalur terpendek
plot_graph_with_shortest_path(graph, positions, shortest_path)
