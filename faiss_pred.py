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
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


analyzer = CountVectorizer().build_analyzer()
stemmer = nltk.stem.PorterStemmer()


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


def preproc(text):
    vectorizer = CountVectorizer(analyzer = stemmed_words,
                                lowercase = True,
                                tokenizer = tokenize,
                                preprocessor = None,
                                stop_words = 'english',
                                ngram_range=(1, 3),
                                max_features = 3500)
    bow_cv = vectorizer.fit(all_texts)
    bow_cv = vectorizer.transform([text])
    tfidf = TfidfTransformer().fit_transform(bow_cv)
    tfidf = tfidf.toarray()
    return tfidf


def inference(row, row_count):
    result = []
    preproc_row = preproc(row)
    D, I = index.search(preproc_row, row_count)
    with open('all_texts.txt', 'r') as f_obj:
        texts = f_obj.read().split('!!!!!')
    for el in I[0]:
        result.append(texts[el])
    return result

