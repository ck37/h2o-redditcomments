import re
from collections import Counter

from nltk import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack


####################################################

'''
Concatenate both feature matrices together
Comments must be a string iterable
Returns sparse matrix
'''
def feature_matrix(comments):
    return hstack( doc_term_matrix(comments),
                   raw_counts_matrix(comments) )

####################################################

'''
Transform into feature matrix
Returns sparse matrix
'''
def doc_term_matrix(comments):
    vectorizer = TfidfVectorizer(
                        analyzer='word',
                        stop_words='english',
                        ngram_range=(1,2),
                        max_features=5000,
                        tokenizer=StemmerTokenizer())

    return vectorizer.fit_transform(comments)

'''
Get raw counts of non-word characters
Returns sparse matrix
'''
def raw_counts_matrix(comments):
    NOT_WORDS = re.compile(r'[^a-zA-Z]')
    counts = lambda s : Counter( re.findall(NOT_WORDS, s) )
    vecs = [counts(c) for c in comments]

    return DictVectorizer().fit_transform(vecs)

####################################################

'''
Custom tokenizer
Performs Porter-Stemming on words
'''
class StemmerTokenizer():
    def __init__(self):
        self.lem = PorterStemmer()

    def __call__(self, string):
        tokens = word_tokenize( string.lower() )
        return [ self.lem.stem_word(t) for t in tokens ]
