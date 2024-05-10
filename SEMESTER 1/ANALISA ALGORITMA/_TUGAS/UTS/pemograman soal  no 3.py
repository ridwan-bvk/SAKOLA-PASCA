def heapify_bottom_up(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify_bottom_up(arr, n, largest)

def build_max_heap(arr):
    n = len(arr)
    # Starting from the last non-leaf node and heapify them in reverse order
    for i in range(n // 2 - 1, -1, -1):
        heapify_bottom_up(arr, n, i)

# Input larik dari pengguna
arr = list(map(int, input("Masukkan elemen: ").split()))

# Membangun heap maksimum
build_max_heap(arr)
print("Heap maksimum:", arr)