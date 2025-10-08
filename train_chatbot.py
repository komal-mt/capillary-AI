import os
import re
import string
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Folder where scraped data is stored
# -------------------------
DATA_DIR = r"E:\capillary\part 2\scraped_docs"

# -------------------------
# Load text from JSON/JSONL files with "paragraphs" key
# -------------------------
documents = []

def clean_text_data(text):
    if not isinstance(text, str):
        return ""
    return text.strip()

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".json") or filename.endswith(".jsonl"):
        file_path = os.path.join(DATA_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Collect text from 'paragraphs' field if it exists
                if "paragraphs" in data and isinstance(data["paragraphs"], list):
                    text = " ".join(data["paragraphs"])
                    text = clean_text_data(text)
                    if text:
                        documents.append(text)
                # fallback if other keys exist
                elif "content" in data:
                    text = clean_text_data(data["content"])
                    if text:
                        documents.append(text)

if not documents:
    print(" No valid text found in JSON files.")
    exit()

print(f" Loaded {len(documents)} text entries from '{DATA_DIR}'")

# -------------------------
# Text Preprocessing
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

corpus = [clean_text(doc) for doc in documents]

# -------------------------
# Build TF-IDF model
# -------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

print(" TF-IDF model built successfully!")

# -------------------------
# Chatbot logic
# -------------------------
def get_response(user_query):
    user_query = clean_text(user_query)
    query_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_match_idx = similarities.argmax()
    score = similarities[best_match_idx]

    if score < 0.1:
        return " Sorry, I couldn't find relevant info in the docs."

    return documents[best_match_idx]

# -------------------------
# Chat Interface
# -------------------------
print("\n Capillary Docs Chatbot Ready!")
print("Type 'exit' to quit.\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye! ")
        break
    response = get_response(query)
    print("Chatbot:", response[:800], "\n") 
