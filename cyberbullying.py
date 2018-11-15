# Data Processing tools
import praw
import pickle
import pandas
import numpy

# Machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Natural language processing tools

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re

class CyberBullyingDetectionEngine:

    ''' Class that deals with training and deploying cyberbullying detection models'''

    def _init_(self):
        self.corpus = None
        self.tags = None
        self.lexicon = None
        self.vectorizer = None
        self.model = None
        self.metrics = None

    def _simplify(self, corpus):
        """ Takes in a list of strings (corpus) and remove stopwords, convert to lowercase
        removes non-letter/number characters (like emojis), stems words so that different tenses are treated the same
        """
        stop_words = set(stopwords.words('english'))
        stemmer = SnowballStemmer('english')

        def clean(text):
            text = re.sub('[^a-zA-Z0-9]', ' ',  text)
            words = [stemmer.stem(w) for w in word_tokenize(text.lower()) if w not in stop_words]
            return " ".join(words)

        return [clean(text) for text in corpus]

    def _model_metrics(self, features, tags):
        """takes in testing data and return a dictionary of metrics"""
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        predictions = self.model.precit(features)
        for r in zip(predictions, tags):
            if r[0] == 1 and r[1]== 1:
                tp += 1
            elif r[0] == 1 and r[1] == 0:
                fp += 1
            elif r[0] == 0 and r[1] == 1:
                fn += 1
            elif r[0] == 0 and r[0] == 0:
                tn += 1


        predictions = self.model.predict(features)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return{
            'precision': precision,
            'recall': recall,
            'f1': (2 * precision * recall) / precision + recall
        }

    def load_corpus(self, path, corpus_col, tag_col):
        """ Takes in a path to pickled pandas dataframe, the name of corpus, the name of corpus column,
        and the name of tag column, and extracts a tagged corpus
        """

        data = pandas.read_pickle(path)[(corpus_col, tag_col)].values
        self.corpus = [row[0] for row in data]
        self.tags = [row[1]for row in data]

    def train_using_bow(self):
        """Trains a model using Bag of Words (word counts) on the corpus and tags"""

        corpus = self._simplify(self.corpus)
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(corpus)

        bag_of_words  = self.vectorizer.transform(corpus)
        x_train, x_test, y_train, y_test = train_test_split(bag_of_words, self.tag, test_size=0.2, stratify=self.tags)

        self.model = MultinomialNB()
        self.model.fit(x_train, y_train)

        self.metrics = self._model_metrics

    def predict(self,corpus):
        """ Takes in a text corpus and returns predictions """

        x = self.vetorizer.transform(self._simplify(corpus))
        return self.model.predict(x)

    def evaluates(self):
        return self.metrics

    
