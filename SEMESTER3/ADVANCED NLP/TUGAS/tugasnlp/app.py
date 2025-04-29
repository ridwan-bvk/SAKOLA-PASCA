from flask import Flask, render_template, request, redirect, url_for, flash
import os

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

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
