# Import library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# 1. Load Dataset
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
url="E:\\pythonAppGado\\bigdata-uas\\OnlineRetail.xlsx"
df = pd.read_excel(url)

# 2. Preprocessing Data
def preprocess_data(df):
    # 2.1. Penanganan missing values
    print("Jumlah missing values sebelum pembersihan:")
    print(df.isnull().sum())
    
    df = df.dropna(subset=['Description', 'CustomerID'])
    df = df[df['Quantity'] > 0]
    
    # 2.2. Filter retur barang (InvoiceNo diawali 'C')
    df = df[~df['InvoiceNo'].astype(str).str.contains('C')]
    
    # 2.3. Konversi Description ke huruf kapital
    df['Description'] = df['Description'].str.strip().str.upper()
    
    # 2.4. Filter negara (opsional: fokus ke satu negara)
    # df = df[df['Country'] == 'United Kingdom']
    
    print("\nJumlah missing values setelah pembersihan:")
    print(df.isnull().sum())
    print(f"\nShape data setelah preprocessing: {df.shape}")
    return df

df = preprocess_data(df)

# 3. Persiapan Data untuk Association Rule Mining
# 3.1. Kelompokkan produk per transaksi
basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'] \
            .sum() \
            .unstack() \
            .reset_index() \
            .fillna(0) \
            .set_index('InvoiceNo')

# 3.2. Konversi ke format boolean (1 = produk dibeli, 0 = tidak)
basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

# 4. Ekstraksi Itemset yang Sering Muncul (FP-Growth)
frequent_itemsets = fpgrowth(basket_sets, min_support=0.02, use_colnames=True)

# 5. Generate Association Rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

# 6. Filter Rules yang Relevan
strong_rules = rules[
    (rules['confidence'] >= 0.5) & 
    (rules['lift'] > 1) &
    (rules['antecedents'].apply(len) == 1) &
    (rules['consequents'].apply(len) == 1)
].copy()

# 7. Format hasil rules
strong_rules['antecedents'] = strong_rules['antecedents'].apply(lambda x: next(iter(x)))
strong_rules['consequents'] = strong_rules['consequents'].apply(lambda x: next(iter(x)))

# 8. Visualisasi Data
plt.figure(figsize=(15, 12))

# 8.1. Top 20 Produk Terpopuler
plt.subplot(2, 2, 1)
top_products = df['Description'].value_counts().nlargest(20)
sns.barplot(y=top_products.index, x=top_products.values, palette="viridis")
plt.title('Top 20 Produk Terpopuler')
plt.xlabel('Jumlah Transaksi')

# 8.2. Heatmap Support vs Confidence
plt.subplot(2, 2, 2)
sns.scatterplot(data=rules, x='support', y='confidence', 
                size='lift', hue='lift', palette="coolwarm", 
                sizes=(20, 200), alpha=0.7)
plt.title('Hubungan Support vs Confidence')
plt.grid(True)

# 8.3. Network Graph Asosiasi Produk
plt.subplot(2, 1, 2)
G = nx.from_pandas_edgelist(
    strong_rules.head(10), 
    source='antecedents', 
    target='consequents',
    edge_attr=['lift', 'confidence'],
    create_using=nx.DiGraph()
)

pos = nx.spring_layout(G, k=0.5)
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1500)
nx.draw_networkx_edges(G, pos, edge_color='gray', width=[r['lift']*0.5 for r in strong_rules.head(10).to_dict('records')], 
                       arrowstyle='->', arrowsize=15)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

edge_labels = {(r['antecedents'], r['consequents']): 
               f"Lift: {r['lift']:.2f}\nConf: {r['confidence']:.2f}" 
               for _, r in strong_rules.head(10).iterrows()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title('Asosiasi Produk (Top 10 Rules)', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig('retail_analysis.png', dpi=300)
plt.show()

# 9. Rekomendasi Bundling Produk
def generate_recommendations(rules_df, top_n=5):
    recommendations = []
    for _, row in rules_df.head(top_n).iterrows():
        rec = {
            'Produk_Utama': row['antecedents'],
            'Bundling': row['consequents'],
            'Confidence': f"{row['confidence']:.2%}",
            'Lift': f"{row['lift']:.2f}",
            'Keterangan': f"Pelanggan yang membeli {row['antecedents']} memiliki kemungkinan {row['confidence']:.2%} untuk membeli {row['consequents']}"
        }
        recommendations.append(rec)
    return pd.DataFrame(recommendations)

# 10. Output Hasil Analisis
print("\nTop 5 Aturan Asosiasi:")
print(strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

print("\nRekomendasi Bundling Produk:")
recommendations = generate_recommendations(strong_rules)
print(recommendations)

# 11. Simpan hasil untuk laporan
strong_rules.head(20).to_csv('association_rules.csv', index=False)
recommendations.to_csv('product_bundling_recommendations.csv', index=False)