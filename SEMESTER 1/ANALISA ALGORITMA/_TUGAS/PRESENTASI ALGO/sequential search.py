def sequential_search(arr, key):
    for i in range(len(arr)):
        if arr[i] == key:
            return i
    return -1

def main():
    n = int(input("Masukkan jumlah elemen dalam array: "))
    arr = []
    for i in range(n):
        num = int(input(f"Masukkan elemen ke-{i+1}: "))
        arr.append(num)
    
    search_key = int(input("Masukkan bilangan yang ingin dicari: "))

    result = sequential_search(arr, search_key)
    if result != -1:
        print(f"Elemen {search_key} ditemukan di indeks {result}.")
    else:
        print(f"Elemen {search_key} tidak ditemukan dalam array.")

if __name__ == "__main__":
    main()
