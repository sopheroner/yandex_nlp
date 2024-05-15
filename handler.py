import faiss
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import nltk
import faiss
import string

df = pd.read_parquet('train-00000-of-00001.parquet')
all_texts = list(set(df['question_1'].to_list() + df['question_2'].to_list()))  # Всего 4567 уникальных вопроса


class FastApiHandler:
    def __init__(self) -> None:
        self.param_types = {
            "user_id": str,
            "model_params": dict
        }
        self.model_path = 'mle_index.index'
        self.load_index_model(self.model_path)
        
    def load_index_model(self, model_path: str):
        try:
            self.index = faiss.read_index(model_path)
        except Exception as e:
            print(f'Failed to load model {e}')

    def preproc_df(self, row):
        def tokenize(text):
            text = ''.join([ch for ch in text if ch not in string.punctuation])
            tokens = nltk.word_tokenize(text)
            lemmatizer = nltk.WordNetLemmatizer()
            return [lemmatizer.lemmatize(token) for token in tokens]
        analyzer = CountVectorizer().build_analyzer()
        stemmer = nltk.stem.PorterStemmer()

        def stemmed_words(doc):
            return (stemmer.stem(w) for w in analyzer(doc))

        vectorizer = CountVectorizer(analyzer=stemmed_words,
                                    lowercase=True,
                                    tokenizer=tokenize,
                                    preprocessor=None,
                                    stop_words='english',
                                    ngram_range=(1, 3),
                                    max_features=3500)
        vectorizer.fit(all_texts)
        bow_cv = vectorizer.transform([row])
        tfidf = TfidfTransformer().fit_transform(bow_cv)
        tfidf = tfidf.toarray()
        return tfidf
    
    def predict_rows(self, index_params):
        result = []
        preproc_row = self.preproc_df(index_params['row'])
        topn = index_params['row_count']
        D, I = self.index.search(preproc_row, topn)
        with open('all_texts.txt', 'r') as f_obj:
            texts = f_obj.read().split('!!!!!')
        for el in I[0]:
            result.append(texts[el])
        return result
    
    def handle(self, params):
        try:
            results = self.predict_rows(params['model_params'])
            response = {
                "results": results
            }
        except Exception as e:
            print(f"Error while handling request: {e}")
        else:
            return response


if __name__ == "__main__":
    # создаём тестовый запрос
    test_params = {"user_id": '123',
                   "model_params": {
                       "row": 'Can I drink alcohol every day?',
                       "row_count": 5
                   }}
    handler = FastApiHandler()
    handler.handle(test_params)
