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
    
    def load_corpus(selfself, model_name):
        """ Takes in a path to pickled pandas dataframe, the name of corpus, the name of corpus,
        """


