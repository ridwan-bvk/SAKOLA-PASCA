import pandas as pd

# Load the uploaded dataset
file_path = r"D:\ABI\__MATA KULIAH PASCA\clone 2\SAKOLA-PASCA\REFERENSI TESIS TOPIC AL-QUR'AN\DATASET ALQURAN\alquran_terjemah_indonesian.csv"
df = pd.read_csv(file_path)

# Tampilkan info ringkas dan beberapa baris awal
df.info(), df.head()

# Ambil kolom penting
df_clean = df[['surah', 'ayah', 'text']].copy()

# Hapus baris dengan nilai kosong di kolom 'text'
df_clean = df_clean.dropna(subset=['text'])

# Gabungkan surah dan ayah menjadi kolom ID ayat
df_clean['ayah_id'] = df_clean['surah'].astype(str) + ':' + df_clean['ayah'].astype(int).astype(str)

# Susun ulang kolom agar rapi
df_clean = df_clean[['ayah_id', 'surah', 'ayah', 'text']]

# Tampilkan beberapa baris hasilnya
df_clean.head()



# Tambahkan kolom baru 'word_count' untuk menghitung jumlah kata pada setiap teks ayat
df_clean['word_count'] = df_clean['text'].apply(lambda x: len(str(x).split()))

# Tampilkan beberapa contoh
df_clean[['ayah_id', 'text', 'word_count']].head()

df.info(), df.head()

df = pd.read_csv(r"D:\ABI\__MATA KULIAH PASCA\clone 2\SAKOLA-PASCA\REFERENSI TESIS TOPIC AL-QUR'AN\DATASET ALQURAN\alquran_terjemah_indonesian_fixrapih.csv")

# Hitung jumlah kata di setiap baris kolom 'text'
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

# Tampilkan hasilnya
print(df[['ayah_id', 'text', 'word_count']].head())

# Jumlahkan semua word_count
total_kata = df['word_count'].sum()
# 71156

print(f"Total jumlah kata dalam seluruh ayat: {total_kata}")

# Simpan dataset yang telah dirapikan ke file CSV baru
# cleaned_file_path = r"D:\ABI\__MATA KULIAH PASCA\clone 2\SAKOLA-PASCA\REFERENSI TESIS TOPIC AL-QUR'AN\DATASET ALQURAN\alquran_terjemah_indonesian_fixrapih.csv"
# df_clean.to_csv(cleaned_file_path, index=False)

# cleaned_file_path

