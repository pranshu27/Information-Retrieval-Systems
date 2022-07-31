import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import sent_tokenize , word_tokenize
import glob
import re
import os
import numpy as np
import sys
import math
from pathlib import Path
from collections import Counter
import numpy as np
Stopwords = set(stopwords.words('english'))
ps=PorterStemmer()
import pickle
import pandas as pd
import sys


def remove_special_characters(text):
    try:
        regex = re.compile('[^a-zA-Z\s]')
        text_returned = re.sub(regex,' ',text)
        return text_returned
    except:
        return text

def tfidf(query, count = 10):

    with open('processed_docs/df.pkl','rb') as file:
        DF=pickle.load(file)
        file.close()

    with open('processed_docs/posting_list.pkl','rb') as file:
        tf=pickle.load(file)
        file.close()
        
    with open('processed_docs/file_idx.pkl','rb') as file:
        file_idx=pickle.load(file)
        file.close()

    text = remove_special_characters(query)
    text = re.sub(re.compile('\d'),'',text)
    words = word_tokenize(text)
    words = [word.lower() for word in words]
    words=[ps.stem(word) for word in words]
    words=[word for word in words if word not in Stopwords]
    words=[word for word in words if word in tf.keys()]


    q=[]
    q_norm=0
    for w in words:
        tf_idf=(words.count(w)*np.log(len(file_idx)/DF[w]))
        q.append(tf_idf)
        q_norm+=tf_idf**2
    q_norm=math.sqrt(q_norm)

    with open('processed_docs/doc_words.pkl','rb') as file:
        doc_words=pickle.load(file)
        file.close()

    with open('processed_docs/doc_norm.pkl','rb') as file:
        doc_norm=pickle.load(file)
        file.close()

    q=np.array(q)/q_norm


    score={}

    for i in range(len(file_idx)):
        doc_v=[]
        for w in words:
            tf_idf=(doc_words[i].count(w)*math.log(len(file_idx)/DF[w]))
            doc_v.append(tf_idf)
        #print(doc_v)
        doc_v=np.array(doc_v)
        score[i]=np.dot(q,doc_v)/doc_norm[i]

    score=sorted(score.items(),key=lambda x:x[1],reverse=True)


    out = []
    
    for i in score:
        if count == 0:
            break
        out.append(file_idx[i[0]])
        #print(file_idx[i[0]],i[1])
        count-=1
    
    return out

query = input("Enter a query:\n")
print("Top 10 relevant DocIDs are as under:")
rank = 1
for file in tfidf(query, 10):
    print("Rank:", str(rank), "DocID:", file)
    rank+=1