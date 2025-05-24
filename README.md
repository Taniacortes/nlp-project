# Suicide Ideation Detection using NLP 
This is a natural language processing (NLP) project where I compare three models to detect suicidal ideation in short texts. The goal is to identify messages that might reflect suicidal thoughts, using different machine learning approaches.


---

## Summary

Using a dataset of over 230,000 Reddit comments, this project applies three binary classification models to determine whether a message expresses suicidal thoughts:

- **Naive Bayes**
- **Logistic Regression**
- **LSTM (Long Short-Term Memory)** with GloVe embeddings

All models were built using Python, with libraries such as `scikit-learn`, `Keras`, `TensorFlow`, and `Gensim`.

---

## Dataset

- Source: Kaggle ([link](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch))
- Language: English
- Labels: `suicide` or `non-suicide`
- Origin: Reddit subreddits like *SuicideWatch*, *depression*, and *teenagers*

---

## Preprocessing

Each text was cleaned and normalized through:

- Lowercasing
- Removing non-letter characters
- Tokenization
- Stopwords removal
- Lemmatization (using `WordNetLemmatizer`)

The processed data was then split into training (80%) and test (20%) sets.

---

## Feature Extraction

- For Naive Bayes and Logistic Regression: **Bag of Words** (CountVectorizer + StandardScaler)
- For LSTM: **GloVe embeddings** (50-dimensional vectors)

---

## Models & Results

| Model               | Feature Extraction | Accuracy |
|--------------------|--------------------|----------|
| Naive Bayes        | Bag of Words       | 90.18%   |
| Logistic Regression| Bag of Words       | 93.39%   |
| LSTM               | GloVe              | 49.82%   |

> Logistic Regression performed best. The LSTM model struggled, possibly due to memory and sequence padding issues.

---

## Testing with Example Inputs

Example sentences were created to test predictions, such as:

```text
"I think the only good solution for me is to end with all my suffering and end my life for good."
