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
        
        if mid_val == key:
            return mid
        elif mid_val < key:
            low = mid + 1
        else:
            high = mid - 1
    
    return -1

def main():
    n = int(input("Masukkan jumlah elemen dalam array: "))
    arr = []
    for i in range(n):
        num = input(f"Masukkan elemen ke-{i+1}: ")
        arr.append(num)
    
    # Pastikan array diurutkan sebelum melakukan binary search
    arr.sort()
    
    search_key = input("Masukkan nilai yang ingin dicari: ")

    result = binary_search(arr, 0, len(arr) - 1, search_key)
    if result != -1:
        print(f"Nilai {search_key} ditemukan di indeks {result}.")
    else:
        print(f"Nilai {search_key} tidak ditemukan dalam array.")

if __name__ == "__main__":
    main()
