import tkinter as tk
from tkinter import Menu
import subprocess

def dfs(graph, start, path, visited, total_distance):
    global shortest_path, shortest_distance
    
    # Mark the current node as visited and add it to the path
    visited.add(start)
    path.append(start)
    
    # If we have visited all nodes, check if the current path is the shortest
    if len(visited) == len(graph):
        if total_distance < shortest_distance:
            shortest_distance = total_distance
            shortest_path = path.copy()
    else:
        # Explore each neighbor
        for neighbor, distance in graph[start].items():
            if neighbor not in visited:
                dfs(graph, neighbor, path, visited, total_distance + distance)
    
    # Backtrack
    visited.remove(start)
    path.pop()

def find_shortest_path():
    global shortest_path, shortest_distance
    shortest_path = []
    shortest_distance = float('inf')
    
    # Define the graph with weights
    graph = {
        'P': {'R': 10, 'S': 9, 'Q': 20},
        'R': {'P': 10, 'S': 17, 'Q': 13},
        'S': {'P': 9, 'R': 17, 'Q': 15},
        'Q': {'P': 20, 'R': 13, 'S': 15}
    }
    
    # Call the DFS function starting from node 'P'
    dfs(graph, 'P', [], set(), 0)
    
    # Display the shortest path and its distance
    result_path.set(f"Shortest path: {' -> '.join(shortest_path)}")
    result_distance.set(f"Shortest distance: {shortest_distance}")
    
    # Highlight the shortest path on the canvas
    highlight_shortest_path()

def highlight_shortest_path():
    # Reset all nodes and edges to original color
    for edge in edges:
        canvas.itemconfig(edge, fill="black")
    for node in nodes:
        canvas.itemconfig(node, fill="lightblue")
    
    # Highlight the nodes and edges in the shortest path
    for i in range(len(shortest_path) - 1):
        node1, node2 = shortest_path[i], shortest_path[i+1]
        edge = edge_map[(node1, node2)]
        canvas.itemconfig(edge, fill="red")
        canvas.itemconfig(nodes[node1], fill="yellow")
        canvas.itemconfig(nodes[node2], fill="yellow")

# Create the main window
root = tk.Tk()
root.title("DFS Shortest Path Finder")

# Create and set variables to display the results
result_path = tk.StringVar()
result_distance = tk.StringVar()

# Create a canvas to draw the graph
canvas = tk.Canvas(root, width=400, height=400)
canvas.pack()

# Define node positions
positions = {
    'P': (100, 50),
    'R': (300, 50),
    'S': (100, 300),
    'Q': (300, 300)
}

# Draw nodes
nodes = {}
for node, (x, y) in positions.items():
    nodes[node] = canvas.create_oval(x-20, y-20, x+20, y+20, fill="lightblue")
    canvas.create_text(x, y, text=node, font=("Arial", 12, "bold"))

# Define edges with weights
edges = []
edge_map = {}
graph = {
    'P': {'R': 10, 'S': 9, 'Q': 20},
    'R': {'P': 10, 'S': 17, 'Q': 13},
    'S': {'P': 9, 'R': 17, 'Q': 15},
    'Q': {'P': 20, 'R': 13, 'S': 15}
}

for node, neighbors in graph.items():
    x1, y1 = positions[node]
    for neighbor, weight in neighbors.items():
        x2, y2 = positions[neighbor]
        edge = canvas.create_line(x1, y1, x2, y2, fill="black")
        edges.append(edge)
        edge_map[(node, neighbor)] = edge
        edge_map[(neighbor, node)] = edge
        canvas.create_text((x1+x2)//2, (y1+y2)//2, text=str(weight), fill="blue")

# Function to return to main application
def kembali_ke_aplikasi():
    try:
        subprocess.Popen(["python", "aplikasiUAS.py"])  # Adjust as per your application launch method
        root.quit()  # Close the main window
    except Exception as e:
        print(f"Error: {e}")

# Menu bar
menu_bar = Menu(root)
root.config(menu=menu_bar)

# File menu
file_menu = Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Open")
file_menu.add_command(label="Save")
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

# Edit menu
edit_menu = Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Edit", menu=edit_menu)
edit_menu.add_command(label="Cut")
edit_menu.add_command(label="Copy")
edit_menu.add_command(label="Paste")

# Frame for main menu
frame_menu_utama = tk.Frame(root, width=400, height=300, padx=20, pady=20)
frame_menu_utama.pack(padx=10, pady=10)

# Title
judul_label = tk.Label(frame_menu_utama, text="HASIL SOAL UAS NO 5", font=("Arial", 14))
judul_label.pack()

# Name and NIM in main menu frame
nama_label_utama = tk.Label(frame_menu_utama, text=f"Nama: Rizki Satriawan Sudarsono\nNIM: 231012050025", font=("Arial", 12))
nama_label_utama.pack(pady=10)

# Button to return to main application
tombol_kembali = tk.Button(frame_menu_utama, text="Kembali ke Aplikasi Utama", command=kembali_ke_aplikasi)
tombol_kembali.pack(pady=10)

# Button to find shortest path
find_path_button = tk.Button(frame_menu_utama, text="Find Shortest Path", command=find_shortest_path)
find_path_button.pack()

# Labels to display results
result_path_label = tk.Label(frame_menu_utama, textvariable=result_path)
result_path_label.pack()

result_distance_label = tk.Label(frame_menu_utama, textvariable=result_distance)
result_distance_label.pack()

# Run the main loop
root.mainloop()
