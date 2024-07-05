import pandas as pd
import numpy as np
from bertopic import BERTopic
import re,string

# 1 select data
df = pd.read_csv('alquran_terjemah_indonesian.csv')
text = df['text']

# 2. PREPROCESSING
# Drop nilai yang kosong 
text = text.fillna('')

# 3. Fungsin untuk prepocessing
def clean_text(text):
    text = text.lower()  # Lower
    text = re.sub(r'\d+', '', text)  # Menghapus angka
    text = re.sub(r'\W', ' ', text)  # Menghapus karakter non-alfanumerik
    text = re.sub(r'\s+', ' ', text)  # Menghapus spasi berlebih
    # text = re(str.maketrans("","",string.punctuation))
    return text.strip()

# 4 panggil fungsi clean text dan simpan divariabel docs
docs = text.apply(clean_text)

# 5. Deklarasikan variabel untuk Membuat model BERTopic
model = BERTopic(verbose=True)

# 6.Melakukan fit dan transformasi dokumen
topics, probabilities = model.fit_transform(docs.tolist())

# 7. Fungsi untuk get topic yang dihasilkan
model.get_topic_info()
# 8 Menampilkan topik array ke 0 indeks 1
model.get_topic(0)

# 9 visualisasi data
# visualisasi topic (intertopic Distance Map)
model.visualize_topics()

# 10 model.visualize_barchart() (topic wors score)
model.visualize_barchart()

# 11 model visualisai hirarki
model.visualize_hierarchy()

# 12 Ctf Idf score 
model.visualize_term_rank()
#13.  similiarity map
model.visualize_heatmap()

