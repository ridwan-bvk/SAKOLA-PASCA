import tkinter as tk
from tkinter import Menu, ttk, Canvas, Frame, Scrollbar, Toplevel
import subprocess
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import networkx as nx
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
    # Load the map image (ganti 'peta.png' dengan path file gambar peta Anda)
    map_img = plt.imread("D:/doc/algoritma analis Presentasi/peta.png")

    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Display the map image
    ax.imshow(map_img)

    # Gambar node (wilayah) dengan warna berdasarkan hasil pewarnaan graf
    for node, pos in positions.items():
        color = color_palette[color_map[node] % len(color_palette)]
        ax.scatter(pos[0], pos[1], s=100, edgecolors='k', facecolors=color, alpha=0.7)

    # Gambar edge
    for edge in G.edges():
        color = color_palette[color_map[edge[0]] % len(color_palette)]
        ax.plot([positions[edge[0]][0], positions[edge[1]][0]], [positions[edge[0]][1], positions[edge[1]][1]], '-', color=color, alpha=0.4, linewidth=0.8)

    # Buat legenda
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=wilayah, markersize=6, markerfacecolor=color_palette[color_map[wilayah] % len(color_palette)])
        for wilayah in wilayahs
    ]

    # Tampilkan legenda di jendela baru
    legend_window = Toplevel(root)
    legend_window.title("Legenda")

    legend_frame = Frame(legend_window)
    legend_frame.pack(fill=tk.BOTH, expand=1)

    legend_canvas = Canvas(legend_frame)
    legend_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    scrollbar = Scrollbar(legend_frame, orient=tk.VERTICAL, command=legend_canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    legend_canvas.configure(yscrollcommand=scrollbar.set)
    legend_canvas.bind('<Configure>', lambda e: legend_canvas.configure(scrollregion=legend_canvas.bbox('all')))

    legend_inner_frame = Frame(legend_canvas)
    legend_canvas.create_window((0, 0), window=legend_inner_frame, anchor='nw')

    for element in legend_elements:
        label = ttk.Label(legend_inner_frame, text=element.get_label(), background=element.get_markerfacecolor())
        label.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    canvas.draw()

# Fungsi untuk menampilkan graf pada peta menggunakan tkinter
def show_graph_on_map():
    # Panggil fungsi untuk menampilkan grafik
    plot_graph_on_map(positions, color_map)

# Fungsi untuk keluar dari aplikasi
def exit_app():
    root.quit()

# Fungsi untuk kembali ke aplikasi utama
def kembali_ke_aplikasi():
    # Ganti dengan logika untuk memulai kembali aplikasi utama (misalnya aplikasiUAS.py)
    try:
        subprocess.Popen(["python", "aplikasiUAS.py"])  # Ganti dengan sesuai dengan cara Anda menjalankan aplikasi utama
        root.quit()  # Menutup jendela utama
    except Exception as e:
        print(f"Error: {e}")

# Membuat jendela utama
root = tk.Tk()
root.title("Aplikasi dengan Menu")

# Membuat menu utama
menu_bar = Menu(root)
root.config(menu=menu_bar)

# Menu File
file_menu = Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Open")
file_menu.add_command(label="Save")
file_menu.add_separator()
file_menu.add_command(label="Exit", command=exit_app)

# Menu Edit
edit_menu = Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Edit", menu=edit_menu)
edit_menu.add_command(label="Cut")
edit_menu.add_command(label="Copy")
edit_menu.add_command(label="Paste")

# Nama dan NIM
nama = "Rizki Satriawan Sudarsono"
nim = "231012050025"

# Frame menu utama
frame_menu_utama = tk.Frame(root, width=400, height=300, padx=20, pady=20)
frame_menu_utama.pack(padx=10, pady=10)

# Judul
judul_label = tk.Label(frame_menu_utama, text="HASIL SOAL UAS NO 4", font=("Arial", 14))
judul_label.pack()

# Nama dan NIM di frame menu utama
nama_label_utama = tk.Label(frame_menu_utama, text=f"Nama: {nama}\nNIM: {nim}", font=("Arial", 12))
nama_label_utama.pack(pady=10)

# Tombol kembali ke aplikasi utama
tombol_kembali = tk.Button(frame_menu_utama, text="Kembali ke Aplikasi Utama", command=kembali_ke_aplikasi)
tombol_kembali.pack(pady=10)

# Tombol untuk menampilkan graf pada peta
tombol_tampilkan_graf = tk.Button(frame_menu_utama, text="Tampilkan Graf pada Peta", command=show_graph_on_map)
tombol_tampilkan_graf.pack(pady=10)

# Menampilkan jendela utama
root.mainloop()
