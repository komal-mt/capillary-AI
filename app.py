from flask import Flask, render_template, request, jsonify
import os, re, string, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

DATA_DIR = r"E:\capillary\part 2\scraped_docs"

# Load all text from Capillary docs
documents = []
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".json") or filename.endswith(".jsonl"):
        with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "paragraphs" in data:
                        text = " ".join(data["paragraphs"]).strip()
                        if text:
                            documents.append(text)
                except json.JSONDecodeError:
                    continue

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

corpus = [clean_text(doc) for doc in documents]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

def get_response(query):
    q = clean_text(query)
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    best = sims.argmax()
    score = sims[best]
    if score < 0.1:
        return " Sorry, I couldn’t find relevant info in the docs."
    return documents[best][:1000]

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_input = data.get("message", "")
    response = get_response(user_input)
    return jsonify({"reply": response})

if __name__ == "__main__":
    print("Starting Flask app...")

    app.run(debug=True, port=5000)
