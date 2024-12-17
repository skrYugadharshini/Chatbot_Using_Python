import random
import json
import pickle
import numpy as pd
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignoreLetterss = ['?', '|' ,'.',',']


for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordlist = nltk.word_tokenize(pattern)
        words.extend(wordlist)
        documents.append((wordlist, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(classes))

classes=sorted(set(classes))
pickle.dump(words, open('words.pkl' , 'wb'))
pickle.dump(classes, open('classes.pkl','wb'))



