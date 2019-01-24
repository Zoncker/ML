from bs4 import BeautifulSoup
import requests
import datetime
import pandas as pd
import numpy as np
from collections import Counter
import nltk
from nltk import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import scipy.sparse
import math


def collect_data():
    page = requests.get('https://lite.cnn.io/en')
    soup = BeautifulSoup(page.content, 'html.parser')
    weblinks = soup.find_all("li")
    pagelinks = []

    for link in weblinks:
        url = link.find('a')
        pagelinks.append('https://lite.cnn.io/'+url.get('href'))

    titles = []
    articles = []
    for href in pagelinks:
        parag_text = []
        page = requests.get(href)
        soup = BeautifulSoup(page.text, 'html.parser')
        art_title = soup.find('h2')
        titles.append(art_title.contents)
        article_text = soup.find_all('p')

        del article_text[-3:]
        for par in article_text:
            text = par.get_text()
            parag_text.append(text)
        articles.append(parag_text + ['---'])

    data = {'Title': titles,
            'PageLink': pagelinks,
            'Article': articles,
            'Date': datetime.datetime.now()}

    news = pd.DataFrame(data=data)
    cols = ['Title', 'PageLink', 'Article', 'Date']
    news = news[cols]

    news.to_json('news_data.json')



def main():
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # collect_data()
    stemmer = SnowballStemmer('english')
    df = pd.read_json('news_data.json')
    tokenizer = TreebankWordTokenizer()
    data_tokens = []
    stop_words = set(stopwords.words('english'))

    for text in df['Article']:
        article_tokens = []
        for sentence in text:
            # res = token.word_tokenize(sentence)
            res = tokenizer.tokenize(sentence.lower())
            filtered_sentence = [w for w in res if w not in stop_words]
            filtered_sentence = [w for w in filtered_sentence if w.isalpha()]
            filtered_sentence = [stemmer.stem(w) for w in filtered_sentence]
            article_tokens.append(filtered_sentence)
        data_tokens.append(article_tokens)

    tra = {'Transformed': data_tokens}
    stem = pd.DataFrame(data=tra)
    col = ['Transformed']
    stem = stem[col]

    def dummy_fun(doc):
        return doc

    # tf_coll = []
    # tf_idf = []
    # word_coll = []
    flat_docs = []
    vectorizer = TfidfVectorizer(analyzer='word',
                                 tokenizer=dummy_fun,
                                 preprocessor=dummy_fun,
                                 token_pattern=None)

    for text in stem['Transformed']:
        flat_list = [item for sublist in text for item in sublist]
        flat_docs.append(flat_list)

        # tf_idf.append(resp)
        # tf_dict = {}
        # wordcount = len(flat_list)
        #
        # word_dict = Counter(flat_list)
        #
        # for word, count in word_dict.items():
        #     tf_dict[word] = count/float(wordcount)
        # print(tf_dict)
        # tf_coll.append(tf_dict)

    # idf_dicts = []
    # n = len(tf_coll)
    #
    # for tf in tf_coll:
    #
    #     idf_dicts.append(tf)

    resp = vectorizer.fit_transform(flat_docs)
    # print(vectorizer.vocabulary_)
    tf = pd.DataFrame(resp.todense())
    svd = TruncatedSVD(n_components=100, n_iter=10, random_state=42, tol=0.0)

    re1 = svd.fit_transform(tf)
    print(re1)


main()
