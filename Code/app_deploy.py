import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# ✅ Required NLTK resources (download once)
nltk.download('stopwords')
nltk.download('wordnet')

# ✅ Streamlit config (must come first)
st.set_page_config(page_title="Email Spam Classifier", layout="centered")

# NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# ✅ Preprocessing function (no word_tokenize)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text)  # remove URLs, mentions, hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation and numbers

    tokens = text.split()  # simpler tokenization

    filtered_tokens = []
    for word in tokens:
        if word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            root = stemmer.stem(lemma)
            filtered_tokens.append(root)

    return " ".join(filtered_tokens)

# ✅ Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("mlp_classifier_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# ✅ Streamlit UI
st.title("📧 Email Spam Classifier")
st.write("Paste your email content below. The model will classify it as **Spam** or **Not Spam**.")

email_text = st.text_area("✉️ Email content", height=300)

if st.button("🔍 Predict"):
    if not email_text.strip():
        st.warning("Please enter some email content.")
    else:
        preprocessed = preprocess_text(email_text)
        vectorized_input = vectorizer.transform([preprocessed])
        prediction = model.predict(vectorized_input)[0]
        label = "Spam" if prediction == 1 else "Not Spam"
        st.success(f"✅ Prediction: **{label}**")
