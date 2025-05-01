from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
# from sklearn.decomposition import TruncatedSVD
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

app = Flask(__name__)
app.secret_key = 'supersecret'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt'}

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def build_inverted_index(texts):
    index = {}
    for doc_id, text in texts.items():
        words = text.lower().split()
        for word in set(words):
            if word not in index:
                index[word] = set()
            index[word].add(doc_id)
    return index

# Route utama
@app.route('/', methods=['GET', 'POST'])
def index():
    uploads = os.listdir(app.config['UPLOAD_FOLDER'])
    search_results = []
    query = None
    selected_files = []

    if request.method == 'POST':
        if 'file' in request.files:  # Upload file
            # Upload route (POST)
            if 'file' not in request.files or request.files['file'].filename == '':
                flash("File tidak boleh kosong!", "danger")
                return redirect(request.url)

            file = request.files['file']
            if file and allowed_file(file.filename):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                
                flash(f"File '{file.filename}' berhasil diupload!", "success")
                return redirect(url_for('index'))

                uploads = os.listdir(app.config['UPLOAD_FOLDER'])
            

        elif 'query' in request.form:  # Search
            query = request.form['query'].strip().lower()
            selected_files = request.form.getlist('selected_files')
            if not query:
                flash("Query tidak boleh kosong!", "danger")
                return redirect(request.url)
            
            # Ambil file yang dipilih untuk pencarian
            if selected_files and query:
                texts = {}
                for idx, fname in enumerate(selected_files, start=1):
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        texts[idx] = f.read()

                inverted_index = build_inverted_index(texts)
                query_words = query.split()
                result_docs = None
                for word in query_words:
                    docs_with_word = inverted_index.get(word, set())
                    result_docs = docs_with_word if result_docs is None else result_docs & docs_with_word

                if result_docs:
                    search_results = []
                    for doc_id in sorted(result_docs):
                        content = texts[doc_id]
                        # Highlight setiap kata query di teks
                        for word in query_words:
                            content = content.replace(word, f"<mark>{word}</mark>")
                            content = content.replace(word.capitalize(), f"<mark>{word.capitalize()}</mark>")
                        search_results.append(content)
        else:
            search_results = []

    return render_template('index.html', uploads=uploads, results=search_results, query=query, selected_files=selected_files)

@app.route('/delete/<filename>', methods=['GET'])
def delete_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    return redirect(url_for('index'))


@app.route('/upload-topic-file', methods=['POST'])
def upload_topic_file():
    if 'file' not in request.files or request.files['file'].filename == '':
        flash("File tidak boleh kosong!", "danger")
        return redirect(request.referrer or url_for('process_topic_model'))

    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    flash(f"File '{filename}' berhasil diupload!", "success")
    return redirect(url_for('process_topic_model'))

@app.route('/topic-model', methods=['GET', 'POST'])
def process_topic_model():
    topics = []
    uploads = os.listdir(app.config['UPLOAD_FOLDER'])

    if request.method == 'POST':
        model_choice = request.form.get('model')
        n_topics = int(request.form.get('n_topics'))
        selected_files = request.form.getlist('selected_files')

        documents = []
        for file in selected_files:
            with open(os.path.join(app.config['UPLOAD_FOLDER'], file), encoding='utf-8') as f:
                documents.append(f.read())

        if model_choice == 'lda':
            # Proses LDA
            vectorizer = CountVectorizer(stop_words='english')
            X = vectorizer.fit_transform(documents)
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
            lda.fit(X)

            terms = vectorizer.get_feature_names_out()
            topics = []
            for idx, topic in enumerate(lda.components_): 
                top_terms = sorted(
                    [(terms[i], topic[i]) for i in topic.argsort()[:-6:-1]], 
                    key=lambda x: x[1], 
                    reverse=True
                )
                formatted_terms = [f"{term} ({weight:.4f})" for term, weight in top_terms]
                topic_str = f"Topik {idx+1}: " + ", ".join(formatted_terms)
                # print(topic_str)  # atau flash(topic_str, "info") jika ingin tampil di browser
                topics.append(topic_str)
                # print(f"Topik  {top_terms}")
        elif model_choice == 'bertopic':
             if len(documents) < 2:
                flash("Model BERTopic membutuhkan minimal 2 dokumen untuk diproses.", "danger")
             else:
                # Proses BERTopic (pastikan sudah diinstall)
                from bertopic import BERTopic
                topic_model = BERTopic()
                topics_bertopic, _ = topic_model.fit_transform(documents)
                topics_data = topic_model.get_topic_info()
                for _, row in topics_data.iterrows():
                    topics.append(f"Topik {row['Topic']}: {row['Name']}")

    return render_template("topic_model.html", uploads=uploads, topics=topics)



if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)

