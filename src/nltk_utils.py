import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from snowballstemmer import TurkishStemmer

turkishStemmer = TurkishStemmer()

def tokenize(sentence):
    return word_tokenize(sentence)

def stem(word):
    return turkishStemmer.stemWord(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    pass

tokenize('Selam, ben kerem. Nasılsın bugün ?')
stem('gidiyorum')