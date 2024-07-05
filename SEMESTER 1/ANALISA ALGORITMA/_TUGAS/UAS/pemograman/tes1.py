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
  1: [(2, 9), (3, 7), (4, 3), (5, 2)],
    2: [(6, 4), (7,2),(8, 1)],
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

# Menentukan tahap setiap simpul (stage_of_node[i] = tahap simpul i)
stage_of_node = {
    '1': 1, 
    '2': 2, 
    '3': 2, 
    '4': 3, 
    '5': 3, 
    '6': 3, 
    '7': 4, 
    '8': 4, 
    '9': 5, 
    '10': 6,
    '11': 7,
    '12': 8
}

source = '1'  # Simpul awal
destination = '12'  # Simpul tujuan
stages = 12  # Jumlah tahap

distance, path = shortest_path_multistage(graph, stages, source, destination)
print("Jarak Terpendek:", distance)
print("Jalur Terpendek:", path)
