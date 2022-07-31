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


def BRS(query, counter=5):
    temp=open('processed_docs/posting_list.pkl',"rb")
    posting_lists=pickle.load(temp)

    temp=open('processed_docs/file_idx.pkl','rb')
    file_index=pickle.load(temp)
    with open('processed_docs/file_idx.pkl',"rb") as temp:
        file_idx=pickle.load(temp)
        temp.close()
        
    unique_words=set(posting_lists.keys())
 
    # clean the query
    
    query = remove_special_characters(query)
    query = word_tokenize(query)
    
    words=[]
    for word in query:
        if len(word)>1:
            word=ps.stem(word)
            if word not in Stopwords:
                words.append(word)

    n=len(file_index)
    word_vector=[]
    word_vector_matrix=[]

    for w in words:
        word_vector=[0]*n
        if w in unique_words:
            for x in posting_lists[w].keys():
                word_vector[x]=1
        word_vector_matrix.append(word_vector)

    aa = len(words)
    aa = aa-1

    for i in range(aa):

        vector1=word_vector_matrix[0]
        vector2=word_vector_matrix[1]

        result=[b1&b2 for b1,b2 in zip(vector1,vector2)]
            
        word_vector_matrix.pop(0)
        word_vector_matrix.pop(0)
            
        word_vector_matrix.insert(0,result)

    final_word_vector=word_vector_matrix[0]
    cnt=0
    files=[]
    for i in final_word_vector:
        if i==1:
            files.append(file_idx[cnt])
            counter-=1
        cnt+=1
        if counter==0:
            break
    return files

query = input("Enter a query:\n")
print("Top relevant DocIDs are as under:")
rank = 1
for file in BRS(query, 10):
    print("DocID:", file)
    rank+=1