class Queue:
    def __init__(self, max_size):
        self.max_size = max_size  # Ukuran maksimum antrian
        self.queue = [None] * max_size  # Inisialisasi array untuk menyimpan elemen antrian
        self.front = -1  # Penanda depan antrian
        self.rear = -1   # Penanda belakang antrian

    def is_empty(self):
        return self.front == -1  # Mengembalikan True jika antrian kosong

    def is_full(self):
        return (self.rear + 1) % self.max_size == self.front  # Mengembalikan True jika antrian penuh

    def enqueue(self, value):
        if self.is_full():  # Periksa apakah antrian penuh
            print("Queue penuh.")
        else:
            if self.is_empty():  # Jika antrian kosong, atur depan ke 0
                self.front = 0
            self.rear = (self.rear + 1) % self.max_size  # Perbarui penanda belakang
            self.queue[self.rear] = value  # Tambahkan elemen ke belakang antrian
            print(f"data masuk antrian: {value}")

    def dequeue(self):
        if self.is_empty():  # Periksa apakah antrian kosong
            print("Queue kosong.")
        else:
            value = self.queue[self.front]  # Ambil elemen dari depan antrian
            if self.front == self.rear:  # Jika hanya ada satu elemen
                self.front = self.rear = -1  # Atur kembali penanda antrian
            else:
                self.front = (self.front + 1) % self.max_size  # Perbarui penanda depan
            print(f"data {value} dihapus dari queue.")

    def display(self):
        if self.is_empty():
            print("Queue kosong.")
        else:
            print("Isi queue:")
            if self.front <= self.rear:
                for i in range(self.front, self.rear + 1):  # Tampilkan elemen dari depan ke belakang
                    print(self.queue[i], end=" ")
            else:
                for i in range(self.front, self.max_size):  # Tampilkan elemen dari depan hingga akhir array
                    print(self.queue[i], end=" ")
                for i in range(0, self.rear + 1):  # Tampilkan elemen dari awal array hingga belakang antrian
                    print(self.queue[i], end=" ")
            print()

# Fungsi utama
def main():
    max_size = 5  # Ubah sesuai kebutuhan
    queue = Queue(max_size)

    while True:
        print("\nPILIH SATU OPERASI BERIKUT INI:")
        print("1. Tambah")
        print("2. Hapus")
        print("3. Tampilkan")
        print("4. Keluar")
        choice = int(input("Pilih Operasi: "))

        if choice == 1:
            value = int(input("Data Inputan: "))
            queue.enqueue(value)  # Panggil metode enqueue
        elif choice == 2:
            queue.dequeue()  # Panggil metode dequeue
        elif choice == 3:
            queue.display()  # Panggil metode display
        elif choice == 4:
            print("Program selesai.")
            break
        else:
            print("Pilihan tidak valid.")

if __name__ == "__main__":
    main()
