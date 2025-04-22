import nltk
import joblib
import sklearn
import streamlit as st
import re
import time
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# NLTK Setup
nltk.download('stopwords')
nltk.download('wordnet')

# Page config
st.set_page_config(
    page_title="Email Spam Filter",
    layout="centered",
    initial_sidebar_state="auto"
)

# 🔵 Neon Blue Animated CSS
st.markdown("""
    <style>
    body {
        background: linear-gradient(-45deg, #000428, #004e92, #001f54, #0f2027);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #e0e0e0;
    }

    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .stTextArea textarea {
        background-color: #0b1d3a !important;
        color: #8ecae6 !important;
        font-family: monospace;
        font-size: 15px;
    }

    .stButton>button {
        background-color: #0077b6;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #00b4d8;
        color: black;
    }

    .flash-success {
        background-color: #003c1d;
        padding: 16px;
        border-radius: 10px;
        margin-top: 20px;
    }

    .flash-fail {
        background-color: #3c001a;
        padding: 16px;
        border-radius: 10px;
        margin-top: 20px;
    }

    .sample-card {
        background-color: #031b3f;
        border-left: 4px solid #00b4d8;
        padding: 12px 20px;
        margin-bottom: 10px;
        border-radius: 8px;
        color: #dcdcdc;
        font-size: 14px;
    }

    h1, h3, h4 {
        color: #90e0ef;
    }

    </style>
""", unsafe_allow_html=True)

# NLP setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    filtered_tokens = []
    for word in tokens:
        if word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            root = stemmer.stem(lemma)
            filtered_tokens.append(root)
    return " ".join(filtered_tokens)

@st.cache_resource
def load_model_and_vectorizer():
    BASE_DIR = os.path.dirname(__file__)
    model_path = os.path.join(BASE_DIR, "mlp_classifier_model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Title and subtitle
st.markdown("<h1 style='text-align: center;'>Email Spam Filter</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color:#caf0f8;'>Enter or copy a message below to classify it as spam or not spam.</p>", unsafe_allow_html=True)

# Input area
st.markdown("#### Email Message")
email_text = st.text_area(" ", height=250)

# Sample Messages
st.markdown("### Example Messages")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Ham Examples")
    st.markdown('<div class="sample-card">Hey, are we still on for lunch today at 1 PM? Let me know!</div>', unsafe_allow_html=True)
    st.markdown('<div class="sample-card">I hope you’re well. I’m reaching out regarding your lease, which still needs to be signed to finalize your file. It’s essential that each resident in the unit completes their signature by the end of the day tomorrow.</div>', unsafe_allow_html=True)

with col2:
    st.markdown("#### Spam Examples")
    st.markdown('<div class="sample-card">Congratulations! You\'ve won a $1000 Walmart gift card. Click here to claim now!</div>', unsafe_allow_html=True)
    st.markdown('<div class="sample-card">URGENT! Your account has been compromised. Reset your password immediately using this link.</div>', unsafe_allow_html=True)

# Prediction
st.markdown("### Classify Message")
if st.button("Predict"):
    if not email_text.strip():
        st.warning("Please enter or paste an email message.")
    else:
        preprocessed = preprocess_text(email_text)
        vectorized_input = vectorizer.transform([preprocessed])
        prediction = model.predict(vectorized_input)[0]

        flash_class = "flash-fail" if prediction == 1 else "flash-success"
        color_text = "Spam Detected" if prediction == 1 else "Message Looks Safe"

        st.markdown(f'<div class="{flash_class}"><h3 style="text-align:center;">{color_text}</h3></div>', unsafe_allow_html=True)
        time.sleep(2)
