def dfs(graph, node, visited, path):
    # Tandai node sebagai dikunjungi
    visited[node] = True
    path.append(node)

    # Telusuri semua tetangga node yang belum dikunjungi
    for next_node in graph[node]:
        if not visited[next_node]:
            dfs(graph, next_node, visited, path)

    return path

# Membangun graf dari gambar
graph = {
    "S": ["P", "Q"],
    "P": ["R"],
    "Q": [],
    "R": []
}

# Inisialisasi variabel
start = "S"
visited = {node: False for node in graph}  # Semua node diinisialisasi sebagai belum dikunjungi
path = []

# Menjalankan DFS
path = dfs(graph, start, visited, path)

# Menampilkan hasil
print("Hasil DFS:", path)
