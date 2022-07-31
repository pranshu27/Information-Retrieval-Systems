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
stop_words = set(stopwords.words('english'))
ps=PorterStemmer()
import pickle
import pandas as pd
import sys


def removeSpecialCharacters(text):
    try:
        regex = re.compile('[^a-zA-Z\s]')
        text_returned = re.sub(regex,' ',text)
        return text_returned
    except:
        return text


def bm25(query, count = 10):

    with open('processed_docs/posting_list.pkl','rb') as file:
        tf=pickle.load(file)
        file.close()
        
    with open('processed_docs/df.pkl','rb') as file:
        DF=pickle.load(file)
        file.close()
        
    with open('processed_docs/file_idx.pkl','rb') as file:
        fileIndex=pickle.load(file)
        file.close()
        
    with open('processed_docs/doc_len.pkl','rb') as file:
        doc_len=pickle.load(file)
        file.close()


    # In[10]:


    k=0
    Ld=doc_len
    N=len(fileIndex)
    for i in Ld:
        k+=Ld[i]
    Lavg=k/N
    Lavg


    # In[11]:


    def IDF(q):
        DF1=0
        if q in DF:
            DF1=DF[q]
        ans=math.log((N-DF1+0.5)/(DF1+0.5))
        return ans

    def score_doc(q):
        q = removeSpecialCharacters(q)
        q = re.sub(re.compile('\d'),'',q)
        allWords = word_tokenize(q)
        allWords = [w.lower() for w in allWords]
        allWords=[ps.stem(w) for w in allWords]
        allWords=[w for w in allWords if w not in stop_words]
        #print(allWords)
        for i in range(len(fileIndex)):
            score[i]=0
            for qi in allWords:
                TF=0
                if qi in tf:
                    if i in tf[qi]:
                        TF=tf[qi][i]
                idf=IDF(qi)
                ans=idf*(k+1)*TF/(TF+k*(1-b+b*(Ld[i]/Lavg)))
                score[i]+=ans
                

    k=1.2
    b=0.75

    score={}
    for i in range(len(fileIndex)):
        score[i]=0
    score_doc(query)
    score=sorted(score.items(),key=lambda item: item[1],reverse=True)
    # print(score)

    out = []
    # count = 10
    for i in score:
        if count == 0:
            break

        out.append(fileIndex[i[0]])
        #print(fileIndex[i[0]],i[1])
        count-=1
    
    return out

query = input("Enter a query:\n")

print("Top 10 relevant DocIDs are as under:")
rank = 1
for file in bm25(query, 10):
    print("Rank:", str(rank), "DocID:", file)
    rank+=1
