from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation


app = Flask(__name__)

# Contoh korpus
documents = [
    "Islam adalah agama yang damai dan penuh kasih sayang.",
    "Shalat lima waktu merupakan kewajiban setiap muslim.",
    "Al-Qur'an adalah kitab suci umat Islam.",
    "Pendidikan sangat penting dalam Islam.",
    "Zakat membantu meringankan beban saudara yang membutuhkan."
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    keyword = request.form['keyword'].lower()
    results = [doc for doc in documents if keyword in doc.lower()]
    return render_template('index.html', search_results=results, keyword=keyword)

@app.route('/topic_model', methods=['POST'])
def topic_model():
    n_topics = int(request.form['n_topics'])

    # Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)

    # LSA
    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    lsa = svd.fit_transform(X)

    terms = vectorizer.get_feature_names_out()
    topics = []
    for i, comp in enumerate(svd.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:5]
        topic_terms = [t[0] for t in sorted_terms]
        topics.append(f"Topik {i+1}: {', '.join(topic_terms)}")

    return render_template('index.html', topics=topics)

if __name__ == '__main__':
    app.run(debug=True)
