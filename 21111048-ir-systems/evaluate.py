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
        
    with open('processed_docs/doc_words.pkl','rb') as file:
        doc_words=pickle.load(file)
        file.close()

    with open('processed_docs/doc_norm.pkl','rb') as file:
        doc_norm=pickle.load(file)
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
        out.append([file_idx[i[0]], i[1]])
        #print(file_idx[i[0]],i[1])
        count-=1
    
    return out

def bm25(query, count = 10):

    with open('processed_docs/posting_list.pkl','rb') as file:
        tf=pickle.load(file)
        file.close()
        
    with open('processed_docs/df.pkl','rb') as file:
        DF=pickle.load(file)
        file.close()
        
    with open('processed_docs/file_idx.pkl','rb') as file:
        file_idx=pickle.load(file)
        file.close()
        
    with open('processed_docs/doc_len.pkl','rb') as file:
        doc_len=pickle.load(file)
        file.close()



    k=0
    Ld=doc_len
    N=len(file_idx)
    for i in Ld:
        k+=Ld[i]
        
    Lavg=k/N
    

    def IDF(q):
        DF1=0
        if q in DF:
            DF1=DF[q]
        ans=math.log((N-DF1+0.5)/(DF1+0.5))
        return ans

    def score_doc(q):
        q = remove_special_characters(q)
        q = re.sub(re.compile('\d'),'',q)
        words = word_tokenize(q)
        words = [word.lower() for word in words]
        words=[ps.stem(word) for word in words]
        words=[word for word in words if word not in Stopwords]
        #print(words)
        for i in range(len(file_idx)):
            score[i]=0
            for qi in words:
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
    for i in range(len(file_idx)):
        score[i]=0
    score_doc(query)
    score=sorted(score.items(),key=lambda item: item[1],reverse=True)
    # print(score)

    out = []
    # count = 10
    for i in score:
        if count == 0:
            break

        out.append([file_idx[i[0]],i[1]])
        #print(file_idx[i[0]],i[1])
        count-=1
    
    return out


query_list=pd.read_csv(sys.argv[1],sep='\t',header=None)
query_list.columns=['qid','query']


with open('processed_docs/file_idx.pkl',"rb") as temp:
    file_idx=pickle.load(temp)
    temp.close()

csv=[]
for index, row in query_list.iterrows():
    # print(row['query'])
    files=BRS(row['query'], 20)
    for file in files:
        csv.append([row['qid'],1,file,1])
    if len(files)<5:
        remaining=[i for i in file_idx.values() if i not in files]
        remaining=remaining[:5-len(files)]
        for file in remaining:
            csv.append([row['qid'],1,file,0])




pd.DataFrame(csv,columns=['QueryID','Iteration','DocId','Relevance']).to_csv('Output/BRS.csv',index=False)
print("Boolean IR System completed.")




csv=[]
for index, row in query_list.iterrows():
    files=tfidf(row['query'], 20)
    for file in files:
        relevance=0
        if file[1]>0:
            relevance=1
        csv.append([row['qid'],1,file[0],relevance])




pd.DataFrame(csv,columns=['QueryID','Iteration','DocId','Relevance']).to_csv('Output/TFIDF.csv',index=False)
print("TF-IDF completed.")




csv=[]
for index, row in query_list.iterrows():
    files=bm25(row['query'], 20)
    for file in files:
        relevance=0
        if file[1]>0:
            relevance=1
        csv.append([row['qid'],1,file[0],relevance])



pd.DataFrame(csv,columns=['QueryID','Iteration','DocId','Relevance']).to_csv('Output/BM25.csv',index=False)
print("BM-25 completed.")
