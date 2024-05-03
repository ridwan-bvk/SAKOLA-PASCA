def pencarian_sequence(dlist, item):
    found = False
    row = None  # Ubah row menjadi None untuk menandai bahwa tidak ada indeks ditemukan

    for index, value in enumerate(dlist):
        if value == item:
            found = True
            row = index
            break  # Keluar dari loop setelah elemen ditemukan

    return found, row

# Input list secara manual
teslis = input("silahkan masukan data dengan pemisah spasi/koma: ")

# Periksa apakah ada spasi atau koma sebagai pemisah
if ' ' in teslis:
    delimiter = ' '
elif ',' in teslis:
    delimiter = ','
else:
    print("Warning: data harus dipisahkan dengan spasi atau koma.")
    exit()

# Pisahkan input menjadi elemen-elemen dan ubah ke dalam list integer
teslis = teslis.split(delimiter)
# teslis = [int(x) for x in teslis]

# Input item yang ingin dicari
item = input("Masukan data yang dicari: ")

# Panggil fungsi pencarian_sequence dengan input manual
found, index = pencarian_sequence(teslis, item)

if found:
    print(f"Item {item} ditemukan pada indeks: {index}")
else:
    print("Item tidak ditemukan dalam list.")