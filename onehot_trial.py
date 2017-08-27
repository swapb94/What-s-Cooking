import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
import json
from nltk.stem.wordnet import WordNetLemmatizer
import re
import collections
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

with open('train.json','r') as f:
    train_data=json.load(f)

train_cuisine=(i['cuisine'] for i in train_data)
train_cuisine=list(train_cuisine)
train_cuisine=np.array(train_cuisine)
print(train_cuisine)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(train_cuisine)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
print(label_encoder.transform(['greek']))
print(onehot_encoder.transform(label_encoder.transform(['greek'])))
