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

lemmatizer=WordNetLemmatizer()

with open("train.json",'r') as f:
    train_data=json.load(f)
with open("test.json",'r') as f:
    test_data=json.load(f)

train_cuisine=[i['cuisine'] for i in train_data]
train_cuisine=list(set(train_cuisine))
values=np.asarray(train_cuisine)

np.save('labels.npy',values)
train_ingredients=[i['ingredients'] for i in train_data]
X_temp=[]
for i in train_ingredients:
    for j in i:
        X_temp.append(j)

lexicon=[]
for ingredients in X_temp:
    ingredient=ingredients.split()
    for ing in ingredient:
        ing=ing.lower()
        ing=lemmatizer.lemmatize(ing)
        lexicon.append(ing)

dic=collections.Counter(lexicon)

#update (Lexicon) with ing having frequency more than 1k
X1=[]
for word in dic:
    #if(dic.get(word)>300&&dic.get(word)<3000):
    if(dic.get(word)>300):
        X1.append(str(word))
lexicon=list(X1)
#save Lexicon
with open('Lexicon.pickle','wb') as f:
    pickle.dump(lexicon,f)
print('Lexicon.pickel created...')

    

def feature_set_train(train_data):
    feature_sets_train=[]
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
        
    for i in train_data:
        X_temp=i['ingredients']
        current_words=[]
        for ingredients in X_temp:
            ingredient=ingredients.split()
            for ing in ingredient:
                ing=lemmatizer.lemmatize(ing.lower())
                current_words.append(ing)
        #print(current_words)
        features=np.zeros(len(lexicon))
        for ing in current_words:
            if ing in lexicon:
                index_val=lexicon.index(ing)
                features[index_val]+=1
        #print(i['cuisine'])
        lbl=onehot_encoder.transform(label_encoder.transform([i['cuisine']]))
        features=list(features)
        lbl=lbl.reshape(20,)
        lbl=list(lbl)
        #print(lbl)

        feature_sets_train.append([features,lbl])
    with open('feature_sets_train.pickle','wb') as f:
        pickle.dump(feature_sets_train,f)
    print('feature_sets_train.pickel created...')


def feature_set_test(test_data):
    feature_sets_test=[]    
    for i in test_data:
        X_temp=i['ingredients']
        current_words=[]
        for ingredients in X_temp:
            ingredient=ingredients.split()
            for ing in ingredient:
                ing=lemmatizer.lemmatize(ing.lower())
                current_words.append(ing)
        #print(current_words)
        features=np.zeros(len(lexicon))
        for ing in current_words:
            if ing in lexicon:
                index_val=lexicon.index(ing)
                features[index_val]+=1
        #features=list(features)
        feature_sets_test.append([features])
    with open('feature_sets_test.pickle','wb') as f:
        pickle.dump(feature_sets_test,f)
    print('feature_sets_test.pickel created...')

feature_set_train(train_data)
feature_set_test(test_data)

