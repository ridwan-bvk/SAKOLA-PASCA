import pandas as pd
import re

# Baca file CSV
df = pd.read_csv('2. al-Baqarah.csv')

# Ganti 'text' sesuai dengan nama kolom teks (otomatis pilih kolom terakhir jika tidak yakin)
kolom_teks = 'text' if 'text' in df.columns else df.columns[-1]

# Fungsi pembersih simbol
def bersihkan_teks(teks):
    teks = re.sub(r'[^\w\s]', '', teks)  # Hapus semua simbol, kecuali huruf/angka dan spasi
    teks = re.sub(r'\s+', ' ', teks)     # Ganti spasi berlebih jadi satu spasi
    return teks.strip()

# Bersihkan dan gabungkan semua teks menjadi satu baris
teks_bersih = df[kolom_teks].dropna().apply(bersihkan_teks)
gabung_satu_baris = ' '.join(teks_bersih.tolist())

# Simpan ke file .txt
with open('output_satu_baris.txt', 'w', encoding='utf-8') as f:
    f.write(gabung_satu_baris)

print("File berhasil disimpan sebagai 'output_satu_baris.txt'")
