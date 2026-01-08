
# 📰 Fake News Detection using NLP, Machine Learning & Deep Learning

## 📌 Project Summary
This project presents an **end-to-end Fake News Detection system** that classifies news articles as **Real** or **Fake** using advanced **Natural Language Processing (NLP)** techniques.  
The solution covers the **complete data science lifecycle** — from dataset acquisition and exploratory data analysis (EDA) to feature engineering, model building, evaluation, and deep learning.

The project demonstrates how **linguistic patterns, text structure, and contextual signals** can be leveraged to combat misinformation at scale.

---

## 🎯 Problem Statement
The rapid spread of fake news across digital platforms poses serious risks to public trust, democracy, and decision-making.  
The goal of this project is to build a **high-accuracy, scalable, and explainable AI model** capable of identifying fake news articles automatically.

---

## 📂 Dataset
- **Source:** Kaggle – Fake News Detection Dataset
- **Files Used:**
  - `True.csv` – Real news articles
  - `Fake.csv` – Fake news articles

### Dataset Details
- Total records after cleaning: **~44,700**
- Features:
  - `title` – Headline of the article
  - `text` – Full article content
  - `subject` – News category
  - `date` – Publication date
  - `label` – Target variable (0 = Real, 1 = Fake)

---

## 🔍 Exploratory Data Analysis (EDA)
Key analyses performed:

### 📊 Target Distribution
- Slight class imbalance (~52% Fake, ~48% Real)
- No aggressive resampling required

### 🧾 Text Length Analysis
- Fake news articles tend to be **shorter**
- Long-form articles (>1000 words) are predominantly **Real News**
- Text length provides moderate predictive signal

### ☁️ Word Cloud Analysis
**Fake News**
- Emotionally charged language
- Personality-driven narratives
- Sensational and opinionated vocabulary

**Real News**
- Formal journalistic tone
- Institutional and source-based references
- Factual and neutral language

### 😊 Sentiment Analysis
- Most articles (both classes) exhibit **neutral sentiment**
- High overlap between Fake and Real News
- Sentiment alone is not a strong discriminator

---

## 🧹 Text Preprocessing Pipeline
- Lowercasing
- URL and HTML tag removal
- Special character and punctuation removal
- Tokenization
- Stopword removal
- Lemmatization
- Clean text generation

---

## 🧠 Feature Engineering
- **TF-IDF Vectorization**
  - Unigrams + Bigrams
  - Top 10,000 features
- Sequence padding for deep learning models
- Engineered features:
  - Cleaned text
  - Text length
  - Sentiment polarity

---

## 🤖 Models Implemented

### 🔹 Machine Learning Models
| Model | Test Accuracy |
|------|--------------|
| Logistic Regression | ~98.7% |
| Multinomial Naive Bayes | ~98.7% |

✔ Fast, interpretable, and strong baselines

---

### 🔹 Deep Learning Model
#### Convolutional Neural Network (CNN)
**Architecture Highlights:**
- Embedding Layer
- Multiple Conv1D layers (kernel sizes: 3, 5, 7)
- Batch Normalization
- Global Max Pooling
- Dense layers with Dropout

**Performance:**
- Training Accuracy: **~99.9%**
- Validation Accuracy: **~99.8%**
- Stable convergence with minimal overfitting

---

## 📈 Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Training vs Validation Loss & Accuracy curves

---

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Libraries & Frameworks:**
  - NumPy, Pandas
  - Scikit-learn
  - NLTK, TextBlob
  - TensorFlow / Keras
  - Matplotlib, Seaborn, Plotly
  - WordCloud
- **Platform:** Jupyter Notebook / Google Colab

---

## 🚀 How to Run the Project
```bash
# Clone the repository
git clone https://github.com/your-username/fake-news-detection.git

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Fake_News_Detection_v2.ipynb
```

---

## 📌 Key Takeaways
- Fake news is typically shorter and more sensational
- Real news is longer, factual, and source-oriented
- CNN significantly outperforms traditional ML models
- Combining EDA + NLP + Deep Learning yields near-perfect performance

---

## 🔮 Future Enhancements
- Transformer-based models (BERT, RoBERTa, DeBERTa)
- Model explainability using SHAP / LIME
- Real-time inference API using FastAPI
- Dashboard integration (Power BI / Tableau)
- Multilingual fake news detection

---

## 👤 Author
**Ratnesh Kumar Satyarthi**  
Data Scientist | NLP | Machine Learning | Deep Learning  

---
