import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, cross_validation
import csv

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from collections import Counter
from tqdm import tqdm
import collections, re
from sklearn.naive_bayes import MultinomialNB
import random
from random import randint
from sklearn.metrics import average_precision_score

from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

import pickle

df=pd.read_csv('test.csv')       #getting file


df=df[['id','comment_text']]



stop_words=set(stopwords.words("english"))
stop_words.add("you")
stop_words.add("and")
stop_words.add("from")
stop_words.add("of")
stop_words.add("i")
stop_words.add("i")
stop_words.add("in")
stop_words.add("out")
stop_words.add("to")
stop_words.add("the")
stop_words.add("she")
stop_words.add("he")
stop_words.add("life")
stop_words.add("actually")
stop_words.add("there")
stop_words.add("make")
stop_words.add("born")
stop_words.add("as")
stop_words.add("it")
stop_words.add("what")
stop_words.add("why")

data=[]
labels=[]
bag=[]


#print(df.head())

def makebag():
    "Tokenize all sentences in the whole data set and make bag of words"
    i=0
    f=[]
    ids=[]
    print("making whole bag")
    with tqdm(total=dimension) as pbar:
        for index, row in df.iterrows():
            if i>=dimension:                #controlling shape of features
                break
            sentence=df.loc[index,'comment_text']
            id=df.loc[index,'id']
            ids.append(id)
            sentence=str(sentence)
            sentence=sentence.lower()
            sentence = sentence.translate(str.maketrans('','',string.punctuation))
            sentence = sentence.replace('\n', ' ').replace('\r', '')
            sublist=[]
            for word in word_tokenize(sentence):
                if word not in stop_words:
                    sublist.append(word)
            f.append(sublist)
            sublist=[]
            i+=1
            pbar.update(1)
    return f,ids


dimension=len(list(df.iterrows()))
#dimension=50            #dimensions of labels and features

i=0
features=[]
bag,ids=makebag()   #made bag of words

ids=np.array(ids)

w2v =Word2Vec.load('word2vec-model')


features=[]

for sentence in bag:
    x=[]
    for word in sentence:
        try:
            x.append(np.array(w2v.wv[word]))        #getting embedding for each word
        except:
            pass
    features.append(np.array(x))

features=np.array(features)

#padding to get fixed length
features=pad_sequences(features,maxlen=15,padding='post',truncating='post')
print(features.shape)
print(ids.shape)

with open('test_features', 'wb') as fp:
    pickle.dump(features, fp)


with open('test_ids', 'wb') as fp:
    pickle.dump(ids, fp)


'''
def custom_tokenize_sentence(df1,bag1):
    "Tokenizing all the sentences"
    i=0
    j=0
    with tqdm(total=len(list(df1.iterrows()))) as p0bar:
        for index, row in df1.iterrows():
            if i>=dimension_test:                #controlling shape of features
                break
            sentence1=df1.loc[index,'comment_text']
            sentence1=str(sentence1)
            sentence1=sentence1.lower()
            sentence1 = sentence1.translate(str.maketrans('','',string.punctuation))
            sentence1 = sentence1.replace('\n', ' ').replace('\r', '')
            #print("SENTENCE: ",sentence)
            sublist=[]
            j=0
            for word in word_tokenize(sentence1):
                sublist.append(word)
            i+=1
            bag1.append(sublist)
            p0bar.update(1)

#bag1 has all features_set for test data

def one_hot_model1(feature_set,bag):
    "creates one hot model of features"
    n=0
    with tqdm(total=len(bag)) as p1bar:
        for feature_sentence in feature_set:              #record has now [["A","sample","sentence"],["blah"]]
            subfeature=[]
            for arr in bag:
                if arr in feature_sentence:          # if word matches with dict place 1
                    subfeature.append(1)
                    continue
                subfeature.append(0)                    #else place 0
            test_features.append(subfeature)
            p1bar.update(1)



#Now test on testing data
df1=pd.read_csv('test.csv')       #getting file
df1.fillna(-99999, inplace=True)     #all NAN are replaced by -99999

#dimension_test=10

dimension_test=len(list(df1.iterrows()))
 
ids=df1[['id'][0]]

bag1=[]
df1=df1[['comment_text']]

custom_tokenize_sentence(df1,bag1)
test_features=[]
one_hot_model1(feature_set,mywords)

arr=[]
i=0
print("Making predictions..")
with tqdm(total=len(list(df1.iterrows()))) as p5bar:
    for index, row in df1.iterrows():
        if i>=dimension_test:                #controlling shape of features
            break
        m=[test_features[i]]
        result1=clf_toxic.predict_proba(m)
        result2=clf_severe_toxic.predict_proba(m)
        result3=clf_obscene.predict_proba(m)
        result4=clf_threat.predict_proba(m)
        result5=clf_insult.predict_proba(m)
        result6=clf_identity_hate.predict_proba(m)
        arr.append([ids[i],np.take(result1,0),np.take(result2,0),np.take(result3,0)
        ,np.take(result4,0),np.take(result5,0),np.take(result6,0)])
        i+=1
        p5bar.update(1)

csvfile = "mysubmission.csv"

i=0
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    if(i==0):
        writer.writerow(["Id","toxic","severe_toxic","obscene","threat"
        ,"insult","identity_hate"])
    i+=1
    writer.writerows(arr)

print("data dumped to file")
'''