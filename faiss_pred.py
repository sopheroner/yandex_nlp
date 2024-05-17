import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
import faiss
import string


df = pd.read_parquet('train-00000-of-00001.parquet')
all_texts = list(set(df['question_1'].to_list() + df['question_2'].to_list()))  # Всего 4567 уникальных вопроса
index = faiss.read_index('mle_index.index')


def tokenize(text):
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stem = nltk.stem.PorterStemmer()
    stem_tokens = [stem.stem(token) for token in tokens]
    return stem_tokens


def preproc(text):
    vectorizer = CountVectorizer(tokenizer=tokenize,
                                 stop_words='english',
                                 max_features=3500)
    vectorizer.fit(all_texts)
    bow_cv = vectorizer.transform([text])
    bow_cv = TfidfTransformer().fit_transform(bow_cv)
    bow_cv = bow_cv.toarray()
    return bow_cv


def inference(row, row_count):
    result = []
    preproc_row = preproc(row)
    D, I = index.search(preproc_row, row_count)
    with open('all_texts.txt', 'r') as f_obj:
        texts = f_obj.read().split('!!!!!')
    for el in I[0]:
        result.append(texts[el])
    return result

