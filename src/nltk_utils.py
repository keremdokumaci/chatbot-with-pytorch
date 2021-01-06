import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from snowballstemmer import TurkishStemmer

turkishStemmer = TurkishStemmer()
stopWords = set(stopwords.words('turkish'))

def tokenize(sentence):
    return word_tokenize(sentence)

def extract_stop_words(tokenized_sentence):
    new_tokenized_sentence = list()
    for word in tokenized_sentence:
        if word not in stopWords:
            new_tokenized_sentence.append(word)
    return new_tokenized_sentence

def stem(word):
    return turkishStemmer.stemWord(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for index,word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[index] = 1.0

    return bag

tokenize('Selam, ben kerem. Nasılsın bugün ?')
stem('gidiyorum')
extract_stop_words(["Selam","ben","ve","Kerem","yarın","size","gelicez"])