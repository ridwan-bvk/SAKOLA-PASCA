import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

# Definisikan wilayah dan atributnya dengan variabel
wilayah_1 = 'batang (1)'
wilayah_2 = 'tegal (2)'
wilayah_3 = 'purwerejo (3)'
wilayah_4 = 'west java (4)'
wilayah_5 = 'demak (5)'
wilayah_6 = 'cilacap (6)'
wilayah_7 = 'Berebes (7)'
wilayah_8 = 'klaten (8)'
wilayah_9 = 'sragen (9)'
wilayah_10 = 'wonogiri (10)'
wilayah_11 = 'Rembang(11)'
wilayah_12 = 'pemalang (12)'
wilayah_13 = 'magelang (13)'
wilayah_14 = 'kudus (14)'
wilayah_15 = 'Sukoharjo (15)'
wilayah_16 = 'semarang (16)'
wilayah_17 = 'Karang Anyar (17)'
wilayah_18 = '18'
wilayah_19 = 'purbalingga (19)'
wilayah_20 = 'pekalongan (20)'
wilayah_21 = 'yogyakarta (21)'
# wilayah_22 = '22'
wilayah_23 = 'semarang (23)'
wilayah_24 = 'Blora (24)'
wilayah_25 = 'salatiga (25)'
wilayah_26 = 'Banyumas (26)'
wilayah_27 = 'temanggung (27)'
wilayah_28 = 'tegal pekal(28)'
wilayah_29 = 'kendal (29)'
wilayah_30 = 'Jepara (30)'
wilayah_31 = 'kebumen (31)'
wilayah_32 = 'Grobogan (32)'
wilayah_33 = '33'
wilayah_34 = '34'
# wilayah_35 = '35'

# Definisikan semua wilayah sebagai daftar
wilayahs = [
    wilayah_1, wilayah_2, wilayah_3, wilayah_4,wilayah_5, wilayah_6, wilayah_7, wilayah_8, wilayah_9, wilayah_10,
    wilayah_11, wilayah_12, wilayah_13, wilayah_14, wilayah_15, wilayah_16, wilayah_17, wilayah_18, wilayah_19, wilayah_20,
    wilayah_21,    wilayah_23, wilayah_24, wilayah_25, wilayah_26, wilayah_27, wilayah_28, wilayah_29, wilayah_30,
    wilayah_31, wilayah_32, wilayah_33, wilayah_34 #, wilayah_35  #   wilayah_22, 
]

# Bangun grafik menggunakan networkx
G = nx.Graph()

# Tambahkan semua wilayah sebagai node
G.add_nodes_from(wilayahs)

# Tentukan posisi node (koordinat untuk plotting)
positions = {
    wilayah_1: (300, 260), wilayah_2: (150, 250), wilayah_3: (300, 100),    wilayah_4: (30, 270),
    wilayah_5: (460, 280), wilayah_6: (100, 140), wilayah_7: (110, 280), wilayah_8: (450, 130),
    wilayah_9: (550, 180), wilayah_10: (530, 50), wilayah_11: (600, 320), wilayah_12: (200, 280),
    wilayah_13: (380, 150), wilayah_14: (490, 300), wilayah_15: (500, 120), wilayah_16: (400, 270),
    wilayah_17: (550, 150), wilayah_18: (250, 180), wilayah_19: (190, 200), wilayah_20: (230, 240),
    wilayah_21: (350, 100),    #  wilayah_22: (660, 110),
    wilayah_23: (410, 230), wilayah_24: (600, 250),    wilayah_25: (420, 200), wilayah_26: (160, 160), wilayah_27: (360, 220), wilayah_28: (270, 290),
    wilayah_29: (340, 270), wilayah_30: (480, 350), wilayah_31: (250, 140), wilayah_32: (495, 250),
    wilayah_33: (430, 150), wilayah_34: (540, 290)#, wilayah_35: (140, 130)
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
    # map_img = plt.imread("peta.png")
    # ax.imshow(map_img, extent=[0, 700, 0, 400])

    # Gambar node (wilayah) dengan warna berdasarkan hasil pewarnaan graf
    for node, pos in positions.items():
        color = color_palette[color_map[node] % len(color_palette)]
        ax.scatter(pos[0], pos[1], s=100, edgecolors='k', facecolors=color, alpha=0.7)
        # Tambahkan nama wilayah di posisi node
        ax.annotate(node, xy=pos, xytext=(5, 5), textcoords='offset points', fontsize=8, color='black')

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
