Thanks for sharing both the notebook and dataset. Based on your notebook (`main.ipynb`) and the dataset (`spam.csv`), you're working on a **Spam Detection Project** using Machine Learning—specifically a binary text classification problem where messages are classified as "spam" or "ham" (not spam).

Here's a clean and structured `README.md` you can include with your project:

---

# 📧 Spam Message Classifier

## 📌 Project Overview

This project aims to build a machine learning model that can classify text messages as **spam** or **ham** (not spam). The model uses natural language processing (NLP) techniques and supervised learning algorithms to detect spam messages based on their content.

This project is developed as part of the final semester coursework for the **Machine Learning course** at George Washington University.

---

## 📁 Dataset

- **File Name**: `spam.csv`
- **Source**: [UCI Machine Learning Repository - SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Columns**:
  - `v1`: Label — `spam` or `ham`
  - `v2`: Text message content

---

## 🧠 Models Used

- **Multinomial Naive Bayes**
- **Logistic Regression**
- **Support Vector Machine (SVM)**

The models are trained on TF-IDF vectorized text data and evaluated based on metrics like **accuracy**, **precision**, **recall**, and **F1-score**.

---

## 🧪 Key Features

- Data cleaning and preprocessing (stopword removal, stemming, lowercasing)
- Text vectorization using **TF-IDF**
- Model evaluation and comparison
- Visualization of results (Confusion Matrix, Accuracy Trends)

---

## 📊 Evaluation Metrics

Each model is evaluated on:
- Accuracy
- Precision
- Recall
- F1-Score

Best-performing model is selected based on these metrics.

---

## 🛠️ Requirements

- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `sklearn`
  - `matplotlib`
  - `seaborn`
  - `nltk`

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. Clone the repository
2. Make sure `spam.csv` is in the working directory
3. Open and run the Jupyter Notebook `main.ipynb`

---

## 👨‍💻 Contributors

- **Aswin Balaji Thippa Ramesh** 
- **Pramod Krishnachari** 
- **Gowri Sriram Lakshmanan**  
  MS in Data Science  
  The George Washington University
