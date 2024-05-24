def sequential_search(arr, key):
    for i in range(len(arr)):
        if arr[i] == key:
            return i
    return -1

def main():
    n = int(input("Masukkan jumlah huruf dalam array: "))
    arr = []
    for i in range(n):
        huruf = input(f"Masukkan huruf ke-{i+1}: ")
        arr.append(huruf)
    
    search_key = input("Masukkan huruf yang ingin dicari: ")

    result = sequential_search(arr, search_key)
    if result != -1:
        print(f"Huruf {search_key} ditemukan di indeks {result}.")
    else:
        print(f"Huruf {search_key} tidak ditemukan dalam array.")

if __name__ == "__main__":
    main()
