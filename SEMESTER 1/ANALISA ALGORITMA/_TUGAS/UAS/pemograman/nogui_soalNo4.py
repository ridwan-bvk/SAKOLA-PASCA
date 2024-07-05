import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

# Definisikan wilayah dan atributnya dengan variabel
wilayah_1 = 'Kabupaten Banjarnegara'
wilayah_2 = 'Kabupaten Banyumas'
wilayah_3 = 'Kabupaten Batang'
wilayah_4 = 'Kabupaten Blora'
wilayah_5 = 'Kabupaten Boyolali'
wilayah_6 = 'Kabupaten Brebes'
wilayah_7 = 'Kabupaten Cilacap'
wilayah_8 = 'Kabupaten Demak'
wilayah_9 = 'Kabupaten Grobogan'
wilayah_10 = 'Kabupaten Jepara'
wilayah_11 = 'Kabupaten Karanganyar'
wilayah_12 = 'Kabupaten Kebumen'
wilayah_13 = 'Kabupaten Kendal'
wilayah_14 = 'Kabupaten Klaten'
wilayah_15 = 'Kabupaten Kudus'
wilayah_16 = 'Kabupaten Magelang'
wilayah_17 = 'Kabupaten Pati'
wilayah_18 = 'Kabupaten Pekalongan'
wilayah_19 = 'Kabupaten Pemalang'
wilayah_20 = 'Kabupaten Purbalingga'
wilayah_21 = 'Kabupaten Purworejo'
wilayah_22 = 'Kabupaten Rembang'
wilayah_23 = 'Kabupaten Semarang'
wilayah_24 = 'Kabupaten Sragen'
wilayah_25 = 'Kabupaten Sukoharjo'
wilayah_26 = 'Kabupaten Tegal'
wilayah_27 = 'Kabupaten Temanggung'
wilayah_28 = 'Kabupaten Wonogiri'
wilayah_29 = 'Kabupaten Wonosobo'
wilayah_30 = 'Kota Yokyakarta'
wilayah_31 = 'Kota Pekalongan'
wilayah_32 = 'Kota Salatiga'
wilayah_33 = 'Kota Semarang'
wilayah_34 = 'Kota Surakarta (Solo)'
wilayah_35 = 'Kota Tegal'

# Definisikan semua wilayah sebagai daftar
wilayahs = [
    wilayah_1, wilayah_2, wilayah_3, wilayah_4, wilayah_5, wilayah_6, wilayah_7, wilayah_8, wilayah_9, wilayah_10,
    wilayah_11, wilayah_12, wilayah_13, wilayah_14, wilayah_15, wilayah_16, wilayah_17, wilayah_18, wilayah_19, wilayah_20,
    wilayah_21, wilayah_22, wilayah_23, wilayah_24, wilayah_25, wilayah_26, wilayah_27, wilayah_28, wilayah_29, wilayah_30,
    wilayah_31, wilayah_32, wilayah_33, wilayah_34, wilayah_35
]

# Bangun grafik menggunakan networkx
G = nx.Graph()

# Tambahkan semua wilayah sebagai node
G.add_nodes_from(wilayahs)

# Tentukan posisi node (koordinat untuk plotting)
positions = {
    wilayah_1: (270, 240), wilayah_2: (150, 250), wilayah_3: (330, 150), wilayah_4: (650, 170),
    wilayah_5: (460, 280), wilayah_6: (100, 140), wilayah_7: (110, 300), wilayah_8: (490, 130),
    wilayah_9: (550, 180), wilayah_10: (530, 50), wilayah_11: (570, 320), wilayah_12: (260, 320),
    wilayah_13: (380, 150), wilayah_14: (490, 300), wilayah_15: (530, 110), wilayah_16: (400, 270),
    wilayah_17: (580, 110), wilayah_18: (250, 180), wilayah_19: (230, 150), wilayah_20: (230, 240),
    wilayah_21: (350, 320), wilayah_22: (660, 110), wilayah_23: (450, 230), wilayah_24: (560, 250),
    wilayah_25: (540, 300), wilayah_26: (160, 160), wilayah_27: (360, 220), wilayah_28: (580, 350),
    wilayah_29: (340, 240), wilayah_30: (450, 370), wilayah_31: (270, 140), wilayah_32: (460, 250),
    wilayah_33: (430, 150), wilayah_34: (540, 290), wilayah_35: (140, 130)
}

# Fungsi untuk menghitung jarak Euclidean antara dua titik
def euclidean_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# Tentukan ambang batas jarak (misalnya 150)
distance_threshold = 150

# Tambahkan edge antara node yang jaraknya kurang dari ambang batas
for node1 in wilayahs:
    for node2 in wilayahs:
        if node1 != node2 and euclidean_distance(positions[node1], positions[node2]) < distance_threshold:
            G.add_edge(node1, node2)

# Algoritma pewarnaan graf greedy
def greedy_graph_coloring(graph):
    color_map = {}
    for node in graph.nodes():
        available_colors = set(range(len(graph.nodes())))
        for neighbor in graph.neighbors(node):
            if neighbor in color_map:
                available_colors.discard(color_map[neighbor])
        color_map[node] = min(available_colors)
    return color_map

# Panggil fungsi pewarnaan graf
color_map = greedy_graph_coloring(G)

# Tentukan warna untuk setiap kelompok berdasarkan hasil pewarnaan graf
color_palette = [
    'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan'
]

# Fungsi untuk plot grafik pada peta
def plot_graph_on_map(positions, color_map):
    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Tambahkan gambar peta di latar belakang plot
    map_img = plt.imread("peta.png")
    ax.imshow(map_img, extent=[0, 700, 0, 400])

    # Gambar node (wilayah) dengan warna berdasarkan hasil pewarnaan graf
    for node, pos in positions.items():
        color = color_palette[color_map[node] % len(color_palette)]
        ax.scatter(pos[0], pos[1], s=100, edgecolors='k', facecolors=color, alpha=0.7)

    # Gambar edge
    for edge in G.edges():
        color = color_palette[color_map[edge[0]] % len(color_palette)]
        ax.plot([positions[edge[0]][0], positions[edge[1]][0]], [positions[edge[0]][1], positions[edge[1]][1]], '-', color=color, alpha=0.4, linewidth=0.8)

    # Buat legenda untuk kelompok warna
    legend_patches = []
    for idx, color in enumerate(color_palette):
        legend_patches.append(mpatches.Patch(color=color, label=f'Group {idx}'))

    # Tambahkan legenda ke plot
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1))

    # Atur batas dan label sumbu
    ax.set_xlim(0, 700)
    ax.set_ylim(0, 400)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Graph Coloring on Map')

    # Tampilkan grafik
    plt.tight_layout()
    plt.show()

# Panggil fungsi untuk menampilkan grafik pada peta
plot_graph_on_map(positions, color_map)
