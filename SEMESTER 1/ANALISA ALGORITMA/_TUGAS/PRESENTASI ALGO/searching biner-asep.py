def pencarian_biner(arr, x):
    low = 0
    high = len(arr) - 1
    
    for i in range(len(arr)):
        mid = (low + high) // 2
        
        # Bandingkan 
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
            
    return -1

# Input list secara manual
arr = input("silahkan masukan data dengan pemisah spasi/koma: ")

# Check if there are spaces or commas as separators
if ' ' in arr:
    delimiter = ' '
elif ',' in arr:
    delimiter = ','
else:
    print("Warning: data harus dipisahkan dengan spasi atau koma!")
    exit()

# Split the input into elements
arr = arr.split(delimiter)

# Input item to search
x = input("Masukan data yang dicari: ")

# Call the binary_search function with manual input
result = pencarian_biner(arr, x)

if result != -1:
    print("Item", x, "ditemukan di index", result)
else:
    print("Item", x, "tidak ditemukan dalam list.")
