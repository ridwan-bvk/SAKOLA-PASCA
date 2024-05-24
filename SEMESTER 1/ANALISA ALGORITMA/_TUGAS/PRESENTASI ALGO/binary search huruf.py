def binary_search_input():
    """
    Fungsi untuk melakukan pencarian biner dalam array yang diinputkan oleh pengguna.
    
    Returns:
        int: Indeks di mana nilai ditemukan, atau -1 jika tidak ditemukan.
    """
    # Input jumlah elemen dari pengguna
    jumlah_elemen = int(input("Masukkan jumlah elemen dalam array: "))
    
    # Input elemen-elemen array dari pengguna
    array = []
    for i in range(jumlah_elemen):
        elemen = input(f"Masukkan elemen ke-{i+1}: ")
        array.append(elemen)
    
    # Urutkan array
    array.sort()
    
    # Input nilai yang ingin dicari dari pengguna
    nilai_cari = input("Masukkan nilai yang ingin dicari: ")
    
    # Tentukan indeks awal dan akhir
    low = 0
    high = jumlah_elemen - 1
    
    # Lakukan binary search
    while low <= high:
        mid = (low + high) // 2
        mid_val = array[mid]
        
        if mid_val == nilai_cari:
            return mid
        elif mid_val < nilai_cari:
            low = mid + 1
        else:
            high = mid - 1
    
    return -1

def main():
    result = binary_search_input()
    if result != -1:
        print(f"Nilai ditemukan di indeks {result}.")
    else:
        print("Nilai tidak ditemukan dalam array.")

if __name__ == "__main__":
    main()
