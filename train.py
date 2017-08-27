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
import pandas as pd
import tensorflow as tf


with open('feature_sets_train.pickle','rb') as f:
    train_data=pickle.load(f)

train_data=np.array(train_data)
train_x=list(train_data[:,0][:-1000])
train_y=list(train_data[:,1][:-1000])

test_x=list(train_data[:,0][-1000:])
test_y=list(train_data[:,1][-1000:])

n_nodes_h1=1900
n_nodes_h2=1000
n_nodes_h3=1200
n_nodes_h4=1500
n_nodes_h5=1500

n_classes=20
batch_size=1000
hm_epochs=10

x=tf.placeholder('float')
y=tf.placeholder('float')

hidden_layer_1={'f_fum':n_nodes_h1,
                'weight':tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_h1])),
                'bias':tf.Variable(tf.random_normal([n_nodes_h1]))}

hidden_layer_2={'f_fum':n_nodes_h2,
                'weight':tf.Variable(tf.random_normal([n_nodes_h1,n_nodes_h2])),
                'bias':tf.Variable(tf.random_normal([n_nodes_h2]))}

hidden_layer_3={'f_fum':n_nodes_h3,
                'weight':tf.Variable(tf.random_normal([n_nodes_h2,n_nodes_h3])),
                'bias':tf.Variable(tf.random_normal([n_nodes_h3]))}

hidden_layer_4={'f_fum':n_nodes_h4,
                'weight':tf.Variable(tf.random_normal([n_nodes_h3,n_nodes_h4])),
                'bias':tf.Variable(tf.random_normal([n_nodes_h4]))}

hidden_layer_5={'f_fum':n_nodes_h5,
                'weight':tf.Variable(tf.random_normal([n_nodes_h4,n_nodes_h5])),
                'bias':tf.Variable(tf.random_normal([n_nodes_h5]))}

output_layer={'f_fum':None,
              'weight':tf.Variable(tf.random_normal([n_nodes_h5,n_classes])),
              'bias':tf.Variable(tf.random_normal([n_classes]))}

def neural_network_model(data):
                l1=tf.add(tf.matmul(data,hidden_layer_1['weight']),hidden_layer_1['bias'])
                l1=tf.nn.relu(l1)

                l2=tf.add(tf.matmul(l1,hidden_layer_2['weight']),hidden_layer_2['bias'])
                l2=tf.nn.relu(l2)
                
                l3=tf.add(tf.matmul(l2,hidden_layer_3['weight']),hidden_layer_3['bias'])
                l3=tf.nn.relu(l3)
                
                l4=tf.add(tf.matmul(l3,hidden_layer_4['weight']),hidden_layer_4['bias'])
                l4=tf.nn.relu(l4)
                
                l5=tf.add(tf.matmul(l4,hidden_layer_5['weight']),hidden_layer_5['bias'])
                l5=tf.nn.relu(l5)
                
    
                output=tf.matmul(l5,output_layer['weight'])+output_layer['bias']

                return output

def train_nn(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(hm_epochs):
                epoch_loss = 0
                i=0
                while i < len(train_x):
                    start = i
                    end = i+batch_size
                    batch_x = np.array(train_x[start:end])
                    batch_y = np.array(train_y[start:end])
                    #print(batch_y)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
                    epoch_loss += c
                    i+=batch_size
                print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))
            print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

train_nn(x)

def predict():
    
