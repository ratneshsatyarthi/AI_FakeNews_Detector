
# 📰 AI-Powered Fake News Detection System

## 📌 Project Overview
This project implements an end-to-end **AI-powered Fake News Detection system** using **Natural Language Processing (NLP)**, **Machine Learning**, and **Deep Learning** techniques.  
The objective is to accurately classify news articles as **Real** or **Fake** by analyzing linguistic patterns, sentiment, and contextual features.

The project includes:
- Comprehensive **Exploratory Data Analysis (EDA)**
- Robust **text preprocessing pipelines**
- Multiple **ML and Deep Learning models**
- Detailed **model evaluation and comparison**

Achieved **~99%+ accuracy** on validation data, demonstrating strong generalization capability.

---

## 🎯 Objectives
- Detect fake news articles with high accuracy
- Understand linguistic differences between real and fake news
- Compare traditional ML models with Deep Learning approaches
- Build a scalable and reusable NLP pipeline

---

## 📂 Dataset
- **Source:** Kaggle – Fake News Detection Dataset  
- **Files:**
  - `True.csv` – Real news articles  
  - `Fake.csv` – Fake news articles  

### Dataset Summary
- Total records (after cleaning): ~44,700
- Features:
  - `title`
  - `text`
  - `subject`
  - `date`
  - `label` (0 = Real, 1 = Fake)

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights from EDA:
- Slight class imbalance (~52% Fake, ~48% Real)
- Fake news articles tend to be **shorter**
- Real news articles are more **formal and source-driven**
- Sentiment polarity is mostly **neutral** for both classes
- Word clouds reveal stylistic differences between Fake and Real news

---

## 🧹 Text Preprocessing
- Lowercasing
- URL & HTML tag removal
- Special character & punctuation removal
- Tokenization
- Stopword removal
- Lemmatization
- Clean text generation

---

## 🧠 Feature Engineering
- **TF-IDF Vectorization**
  - Unigrams + Bigrams
  - Top 10,000 features
- Sequence padding for Deep Learning models
- Derived features:
  - Text length
  - Sentiment polarity

---

## 🤖 Models Implemented

### 🔹 Machine Learning Models
| Model | Test Accuracy |
|------|--------------|
| Logistic Regression | ~98.7% |
| Multinomial Naive Bayes | ~98.7% |

### 🔹 Deep Learning Model
**Convolutional Neural Network (CNN)**
- Embedding layer
- Multiple Conv1D layers (kernel sizes: 3, 5, 7)
- Batch Normalization
- Global Max Pooling
- Dense layers with Dropout

**Performance:**
- Training Accuracy: ~99.9%
- Validation Accuracy: ~99.8%
- Minimal overfitting

---

## 📈 Model Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Training vs Validation curves

---

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:**  
  NumPy, Pandas, Scikit-learn, NLTK, TextBlob, TensorFlow/Keras,  
  Matplotlib, Seaborn, Plotly, WordCloud
- **Platform:** Google Colab / Jupyter Notebook

---

## 🚀 How to Run
```bash
# Clone the repository
git clone https://github.com/your-username/fake-news-detection.git

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Fake_News_Detection.ipynb
```

---

## 📌 Key Takeaways
- Fake news is generally shorter and more sensational
- Real news is factual, longer, and source-oriented
- CNN significantly outperforms traditional ML models
- Combining EDA + NLP + Deep Learning yields near-perfect results

---

## 🔮 Future Enhancements
- Transformer-based models (BERT, RoBERTa)
- Model explainability using SHAP / LIME
- Real-time inference API (FastAPI)
- Dashboard integration (Power BI / Tableau)
- Multilingual fake news detection

---

## 👤 Author
**Ratnesh Kumar Satyarthi**  
Data Scientist | NLP | Deep Learning | BI  

---
