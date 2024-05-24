import pandas as pd

def sequential_search(arr, key):
    """
    Fungsi untuk melakukan pencarian berurutan dalam array.
    
    Parameters:
        arr (list): Array yang akan dicari.
        key (int): Nilai yang ingin dicari dalam array.
        
    Returns:
        int: Indeks pertama di mana nilai ditemukan, atau -1 jika tidak ditemukan.
    """
    for i in range(len(arr)):
        # Bandingkan elemen pada indeks saat ini dengan nilai yang dicari
        if arr[i] == key:
            return i  # Kembalikan indeks jika nilai ditemukan
    return -1  # Kembalikan -1 jika nilai tidak ditemukan

def main():
    # Baca file Excel
    file_path = "C:\Users\riski\algoritma analis\datakaryawan.xlsx"  # Ganti dengan path file Excel Anda
    df = pd.read_excel(file_path)
    
    # Tampilkan data
    print("Data dari file Excel:")
    print(df)
    
    # Ambil kolom yang ingin Anda cari
    column_name = input("Masukkan nama kolom yang ingin Anda cari: ")
    arr = df[column_name].tolist()
    
    # Input nilai yang ingin dicari dari pengguna
    key = input("Masukkan nilai yang ingin dicari: ")
    
    # Lakukan pencarian
    result = sequential_search(arr, key)
    
    # Cetak hasil
    if result != -1:
        print(f"Nilai {key} ditemukan di baris {result + 2}.")  # +2 karena indeks dimulai dari 0 dan baris dimulai dari 2
    else:
        print(f"Nilai {key} tidak ditemukan dalam kolom {column_name}.")

if __name__ == "__main__":
    main()
