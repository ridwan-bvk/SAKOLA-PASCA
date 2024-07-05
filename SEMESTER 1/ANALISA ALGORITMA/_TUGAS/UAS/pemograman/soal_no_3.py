import tkinter as tk
from tkinter import Menu
import subprocess
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

# Fungsi untuk menampilkan GUI
def show_gui():
    # Fungsi untuk menampilkan hasil traversal
    def show_traversal(traversal_type):
        if traversal_type == "Level Order":
            result = level_order_traversal(1)
            explanation = "Level order traversal visits nodes level by level from left to right. It uses BFS algorithm."
        elif traversal_type == "Pre Order":
            result = pre_order_traversal(1)
            explanation = "Pre order traversal visits the root node first, then its children recursively."
        elif traversal_type == "In Order":
            result = in_order_traversal(1)
            explanation = "In order traversal visits the left subtree, then the root, then the right subtree."
        elif traversal_type == "Post Order":
            result = post_order_traversal(1)
            explanation = "Post order traversal visits the children of a node recursively before the node itself."
        
        # Mengupdate label hasil traversal dan penjelasan
        traversal_result_label.config(text=f"{traversal_type} Traversal: {result}")
        traversal_explanation_label.config(text=explanation)

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
    root.title("Traversal Graph GUI")

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

    # Frame utama untuk konten aplikasi
    frame_main = tk.Frame(root)
    frame_main.pack(padx=10, pady=10)

    # Judul
    judul_label = tk.Label(frame_main, text="HASIL SOAL UAS NO 3", font=("Arial", 14))
    judul_label.pack()

    # Nama dan NIM
    nama = "Rizki Satriawan Sudarsono"
    nim = "231012050025"
    nama_label = tk.Label(frame_main, text=f"Nama: {nama}\nNIM: {nim}", font=("Arial", 12))
    nama_label.pack(pady=10)

    # Frame untuk menampilkan graf
    frame_graph = tk.Frame(frame_main)
    frame_graph.pack(side=tk.LEFT, padx=10)

    # Menggambar graf
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=12, font_weight='bold', arrows=True, ax=ax)
    ax.set_title('Traversal Graph')
    canvas = FigureCanvasTkAgg(fig, master=frame_graph)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Frame untuk tombol dan hasil traversal
    frame_controls = tk.Frame(frame_main)
    frame_controls.pack(side=tk.RIGHT, padx=10)

    # Label untuk hasil traversal
    traversal_result_label = tk.Label(frame_controls, text="", font=("Helvetica", 12))
    traversal_result_label.pack(pady=10)

    # Label untuk penjelasan traversal
    traversal_explanation_label = tk.Label(frame_controls, text="", font=("Helvetica", 10), wraplength=250, justify="left")
    traversal_explanation_label.pack(pady=10)

    # Tombol untuk melakukan traversal
    btn_level_order = tk.Button(frame_controls, text="Level Order", command=lambda: show_traversal("Level Order"))
    btn_level_order.pack(pady=5)

    btn_pre_order = tk.Button(frame_controls, text="Pre Order", command=lambda: show_traversal("Pre Order"))
    btn_pre_order.pack(pady=5)

    btn_in_order = tk.Button(frame_controls, text="In Order", command=lambda: show_traversal("In Order"))
    btn_in_order.pack(pady=5)

    btn_post_order = tk.Button(frame_controls, text="Post Order", command=lambda: show_traversal("Post Order"))
    btn_post_order.pack(pady=5)

    # Tombol untuk kembali ke aplikasi utama
    btn_kembali = tk.Button(frame_main, text="Kembali ke Aplikasi Utama", command=kembali_ke_aplikasi)
    btn_kembali.pack(pady=10)

    # Menampilkan jendela utama
    root.mainloop()

# Memanggil fungsi untuk menampilkan GUI
show_gui()
