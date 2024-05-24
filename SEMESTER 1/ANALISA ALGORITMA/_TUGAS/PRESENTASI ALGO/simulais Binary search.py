import pandas as pd

def binary_search(arr, low, high, key):
    """
    Fungsi untuk melakukan pencarian biner dalam array terurut.
    
    Parameters:
        arr (list): Array yang akan dicari.
        low (int): Indeks awal dari array.
        high (int): Indeks akhir dari array.
        key (str atau int): Nilai yang ingin dicari dalam array.
        
    Returns:
        int: Indeks di mana nilai ditemukan, atau -1 jika tidak ditemukan.
    """
    while low <= high:
        mid = (low + high) // 2
        mid_val = arr[mid]
        
        if str(mid_val).lower() == str(key).lower():
            return mid
        elif str(mid_val).lower() < str(key).lower():
            low = mid + 1
        else:
            high = mid - 1
    
    return -1

def main():
    # Baca file Excel
    file_path = "C:/Users/riski/algoritmaanalis/datakaryawan.xlsx"  # Ganti dengan path file Excel Anda
    df = pd.read_excel(file_path)
    
    # Cetak data yang sudah diurutkan
    print("Data dari file Excel:")
    print(df)

    # Ambil kolom yang ingin Anda cari
    column_name = input("Masukkan nama kolom yang ingin Anda urutkan dan cari: ")
    
    # Urutkan DataFrame berdasarkan kolom yang dipilih
    df_sorted = df.sort_values(by=column_name)
    
    # Cetak data yang sudah diurutkan
    print("Data dari file Excel setelah diurutkan:")
    print(df_sorted)
    
    # Ambil kolom yang sudah diurutkan dan konversi menjadi array
    arr = df_sorted[column_name].tolist()
    
    # Input nilai yang ingin dicari dari pengguna
    key = input("Masukkan nilai yang ingin dicari: ")
    
    # Tentukan indeks awal dan akhir
    low = 0
    high = len(arr) - 1
    
    # Lakukan pencarian menggunakan binary search
    result = binary_search(arr, low, high, key)
    
    # Cetak hasil pencarian
    if result != -1:
        print(f"Nilai {key} ditemukan di Index {result}.") 
    else:
        print(f"Nilai {key} tidak ditemukan dalam kolom {column_name}.")

if __name__ == "__main__":
    main()
