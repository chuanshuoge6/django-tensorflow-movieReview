from django.shortcuts import render
from django.http import  HttpResponse
import tensorflow as td
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb
word_index = data.get_word_index()

def review_encode(s):
    #1 means 'start line'
    encoded = [1]

    for word in s:
         #translate human words to machine digits
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            #2 means 'word not found in dictionary'
            encoded.append(2)

    return encoded

def index(request):
    p = ''
    w = ''
    model = keras.models.load_model('model.h5')
    # assume review is either negative or positive
    class_names = ['negative', 'positive']
    try:
        review_posted = request.POST['review']
    except Exception:
        pass
    else:
        nline = review_posted.replace(',', '').replace('(', '').replace(')', '').replace(':', '')\
            .replace("\"",'').strip().split(' ')
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=0, padding='post', maxlen=256)
        w = encode
        predict = model.predict(encode)
        print(review_posted)
        print(encode)
        result = class_names[int(round(predict[0][0]))]
        p = str(predict[0][0]) + ' ' + result
        print(predict[0], result)

    return render(request, 'index.html', {'prediction': p, 'wordMap': w})