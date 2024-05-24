def shortest_path_multistage(graph, stages, source, destination):
    # Inisialisasi jarak terpendek untuk setiap simpul dengan tak hingga
    distances = {node: float('inf') for node in graph}
    # Jarak dari sumber ke sumber adalah 0
    distances[source] = 0

    # Inisialisasi list untuk menyimpan jalur
    path = {node: None for node in graph}

    # Proses dari stage ke stage
    for stage in range(1, stages + 1):
        for u in graph:
            # Jika u ada di stage saat ini
            if stage_of_node[u] == stage:
                for v, cost in graph[u]:
                    if distances[v] > distances[u] + cost:
                        distances[v] = distances[u] + cost
                        path[v] = u

    # Membangun jalur terpendek dari source ke destination
    shortest_path = []
    current = destination
    while current is not None:
        shortest_path.append(current)
        current = path[current]
    shortest_path.reverse()

    return distances[destination], shortest_path

# Contoh penggunaan fungsi
# Representasi graf dalam bentuk adjacency list
graph = {
    'a': [('b', 7), ('c', 6),('d',9)],
    'b': [('e', 6), ('f', 8),('g',12)],
    'c': [('f', 6),('g', 3),('h', 9)],
    'd': [('g', 10),('h', 15),],
    'e': [('i', 5), ('j', 6),('k', 7),],
    'f': [('i', 6),('j', 9),('k', 17)],
    'g': [('j', 4),('k', 9)],
    'h': [('j', 7), ('k', 13)],
    'i': [('l', 7)],
    'j': [('l', 5)],
    'k':[('l', 8)],
    'l':[]
}

# Menentukan tahap setiap simpul (stage_of_node[i] = tahap simpul i)
stage_of_node = {
    'a': 1, 
    'b': 2, 
    'c': 2, 
    'd': 3, 
    'e': 3, 
    'f': 3, 
    'g': 4, 
    'h': 4, 
    'i': 5, 
    'j': 6,
    'k': 7,
    'l': 8
}

source = 'a'  # Simpul awal
destination = 'l'  # Simpul tujuan
stages = 8  # Jumlah tahap

distance, path = shortest_path_multistage(graph, stages, source, destination)
print("Jarak Terpendek:", distance)
print("Jalur Terpendek:", path)
