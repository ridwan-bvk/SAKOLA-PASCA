class Stack:
    def __init__(stack, max_size):
        stack.max_size = max_size  # Menyimpan ukuran maksimum stack
        stack.items = [None] * max_size  # Membuat list kosong dengan ukuran maksimum yang diberikan
        stack.top = -1  # Menyimpan indeks dari item teratas pada stack. Diinisialisasi sebagai -1 karena stack kosong.

    def is_empty(stack):
        return stack.top == -1  # Mengembalikan True jika top sama dengan -1, yang berarti stack kosong.

    def is_full(stack):
        return stack.top == stack.max_size - 1  # Mengembalikan True jika top sama dengan max_size - 1, yang berarti stack penuh.

    def push(stack, item):
        # Memeriksa apakah stack penuh
        if stack.is_full():
            print("Stack penuh, tidak bisa menambahkan item baru.")
            return False

        stack.top += 1  # Menambahkan nilai top untuk menunjuk ke posisi baru pada stack.
        stack.items[stack.top] = item  # Menambahkan item ke posisi baru pada stack.
        stack.peek()  # Memanggil peek setelah push dilakukan untuk menampilkan item teratas.
        return True  # Mengembalikan True karena operasi push berhasil.

    def pop(stack, index=None):
        # Memeriksa apakah stack kosong
        if stack.is_empty():
            print("Stack kosong, tidak bisa melakukan operasi pop.")
            return None

        # Jika indeks tidak ditentukan, pop item teratas
        if index is None:
            popped_item = stack.items[stack.top]
            stack.items[stack.top] = None
            stack.top -= 1
            stack.peek()  # Memanggil peek setelah pop dilakukan untuk menampilkan item teratas.
            return popped_item

        # Jika indeks ditentukan, pop item sesuai dengan indeks yang diberikan
        else:
            if index < 0 or index > stack.top:
                print("Indeks tidak valid.")
                return None
            popped_item = stack.items[index]
            for i in range(index, stack.top):
                stack.items[i] = stack.items[i + 1]
            stack.items[stack.top] = stack
            stack.top -= 1
            stack.peek()  # Memanggil peek setelah pop dilakukan untuk menampilkan item teratas.
            return popped_item

    def peek(stack):
        # Memeriksa item teratas dalam stack
        if stack.is_empty():
            print("Stack kosong, tidak ada item untuk dilihat.")
        else:
            print(f"Item paling atas pada stack: {stack.items[stack.top]}")

    def display(stack):
        # Menampilkan semua data dalam stack
        if stack.top == -1:
            print("Stack kosong.")
        else:
            print("Isi stack:", stack.items[:stack.top + 1])

    def empty(stack):
        # Mengosongkan stack
        stack.items = [None] * stack.max_size
        stack.top = -1
        print("Stack telah dikosongkan.")

def pop_value(stack, value):
    if stack.is_empty():
        print("Stack kosong, tidak bisa melakukan operasi pop.")
        return None

    index = None
    for i in range(stack.top, -1, -1):
        if stack.items[i] == value:
            index = i
            break

    if index is None:
        print(f"Nilai {value} tidak ditemukan dalam stack.")
        return None

    popped_items = []
    while stack.top >= index:
        popped_item = stack.pop()
        popped_items.append(popped_item)

    return popped_items

# Contoh penggunaan untuk operasi push, pop, dan peek dengan input dari pengguna
max_size = int(input("Masukkan ukuran maksimum stack: "))
stack = Stack(max_size)

while True:
    print("'push' untuk menambahkan item ke stack")
    print("'pop' untuk menghapus item dari stack")
    print("'empty' untuk mengosongkan stack")
    print("'exit' untuk keluar")
    choice = input("silahkan ketik command: ")
    if choice.lower() == 'exit':
        break
    elif choice.lower() == 'push':
        item = input("Masukkan item untuk ditambahkan ke stack: ")
        if not item.isdigit():
            print("Masukkan harus berupa angka.")
            continue
        stack.push(int(item))
    elif choice.lower() == 'pop':
        item_to_pop = input("Masukkan nilai yang ingin dihapus dari stack: ")
        if not item_to_pop.isdigit():
            print("Masukkan harus berupa angka.")
            continue
        popped_items = pop_value(stack, int(item_to_pop))
        if popped_items is not None:
            print(f"Nilai {item_to_pop} telah dihapus dari stack.")
            print("Nilai-nilai yang di-pop:", popped_items)
    elif choice.lower() == 'empty':
        stack.empty()
    stack.display()

print("Keluar dari program.")
