import tkinter as tk
from tkinter import Menu, messagebox
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# Global lists to store Entry widgets for Pi and Di
entries_pi = []
entries_di = []
# Membuat jendela utama
root = tk.Tk()
root.title("Aplikasi dengan Menu")
   
# Membuat menu utama
menu_bar = Menu(root)
root.config(menu=menu_bar)

def maximize_profit(n, activities):
    Pi = [activities[x][0] for x in range(1, n + 1)]
    Di = [activities[x][1] for x in range(1, n + 1)]
    
    activities_with_values = list(zip(range(1, n + 1), Pi, Di))
    activities_with_values.sort(key=lambda x: x[1] / x[2], reverse=True)
    
    max_profit = 0
    current_time = 0
    chosen_activities = []
    chosen_activities_indices = []
    cumulative_profit = []
    time_accumulation = 0
    
    for activity in activities_with_values:
        if current_time + activity[2] <= 50:
            chosen_activities.append(activity[0])
            chosen_activities_indices.append(activity[0] - 1)  # Indexing starts from 0 for plotting
            max_profit += activity[1]
            current_time += activity[2]
            cumulative_profit.append(max_profit)
            time_accumulation += activity[2]
        else:
            break
    
    result_activities.config(text=f"Urutan aktivitas untuk keuntungan maksimum: {chosen_activities}")
    result_profit.config(text=f"Keuntungan maksimum yang dapat diperoleh: {max_profit}")
    
    # Plotting cumulative profit over time or activities
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(cumulative_profit)) + 1, cumulative_profit, marker='o', linestyle='-', color='b', label='Cumulative Profit')
    plt.xlabel('Number of Activities')
    plt.ylabel('Cumulative Profit')
    plt.title('Cumulative Profit over Selected Activities')
    plt.xticks(np.arange(len(cumulative_profit)) + 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def menu_action():
    print("Menu action executed!")

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

# Fungsi untuk membuat GUI
def create_gui():
    def start_simulation():
        try:
            n = int(entry_activities.get())  # Mendapatkan jumlah jenis aktivitas dari input pengguna
            activities = {}
            
            for x in range(1, n + 1):
                p = int(entries_pi[x-1].get())
                d = int(entries_di[x-1].get())
                activities[x] = (p, d)
                
            maximize_profit(n, activities)
        
        except ValueError:
            messagebox.showerror("Error", "Masukkan hanya angka untuk Pi dan Di.")
    
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
    judul_label = tk.Label(frame_menu_utama, text="HASIL SOAL UAS NO 1", font=("Arial", 14))
    judul_label.pack()
    
    # Nama dan NIM di frame menu utama
    nama_label_utama = tk.Label(frame_menu_utama, text=f"Nama: {nama}\nNIM: {nim}", font=("Arial", 12))
    nama_label_utama.pack(pady=10)
    
  
    # Label dan Entry untuk jumlah jenis aktivitas
    label_activities = tk.Label(root, text="Masukkan jumlah jenis aktivitas:")
    label_activities.pack()
    entry_activities = tk.Entry(root)
    entry_activities.pack()
    
    # Tombol untuk menambahkan input untuk nilai Pi dan Di
    def add_entries():
        try:
            n = int(entry_activities.get())
            
            # Hapus input yang lama (jika ada)
            for entry in frame_values.winfo_children():
                entry.destroy()
            
            entries_pi.clear()
            entries_di.clear()
            
            # Tambahkan input baru untuk nilai Pi dan Di
            for i in range(n):
                label_pi = tk.Label(frame_values, text=f"Pi untuk jenis aktivitas {i+1}:")
                label_pi.grid(row=i, column=0, padx=5, pady=5)
                entry_pi = tk.Entry(frame_values)
                entry_pi.grid(row=i, column=1, padx=5, pady=5)
                entries_pi.append(entry_pi)
                
                label_di = tk.Label(frame_values, text=f"Di untuk jenis aktivitas {i+1}:")
                label_di.grid(row=i, column=2, padx=5, pady=5)
                entry_di = tk.Entry(frame_values)
                entry_di.grid(row=i, column=3, padx=5, pady=5)
                entries_di.append(entry_di)
        
        except ValueError:
            messagebox.showerror("Error", "Masukkan hanya angka untuk jumlah jenis aktivitas.")
    
    button_add_entries = tk.Button(root, text="Tambahkan nilai Pi dan Di", command=add_entries)
    button_add_entries.pack()
    
    # Frame untuk menampilkan input nilai Pi dan Di
    frame_values = tk.Frame(root)
    frame_values.pack()
    
    # Tombol untuk memulai simulasi
    button_simulate = tk.Button(root, text="Simulasi", command=start_simulation)
    button_simulate.pack()
    
    # Label untuk menampilkan hasil
    global result_activities, result_profit
    result_activities = tk.Label(root, text="")
    result_activities.pack()
    result_profit = tk.Label(root, text="")
    result_profit.pack()
    # Tombol kembali ke aplikasi utama
    tombol_kembali = tk.Button(frame_menu_utama, text="Kembali ke Aplikasi Utama", command=kembali_ke_aplikasi)
    tombol_kembali.pack(pady=10)
    root.mainloop()
 
    
# Panggil fungsi untuk membuat GUI

create_gui()
