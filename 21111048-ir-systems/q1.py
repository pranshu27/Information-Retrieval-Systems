
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
import numpy as nap
Stopwords = set(stopwords.words('english'))
ps=PorterStemmer()
import pickle



from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)



file_folder = 'english-corpora/*'
ps = PorterStemmer()
all_words = []
doc_words=[]  # doc_words is a list of list for stemmed words for every doc
doc_word_count=[] # doc_word_count list of dict containing word:freq for every doc
dict_global = {} # update the word:frequency value in global dictionary
dict_global_test={}
Ld={}
files_with_index = {} #indexing the files with integers for ease of use later on
i=0
idx = 0
for file in glob.glob(file_folder):
    i=i+1
    #p
    # rint(i,"----",file)
    fname = file
    file = open(file , "r",encoding='UTF-8')
    text = file.read()
    text = re.sub('[^a-zA-Z\s]',' ',text)
    tokens = word_tokenize(text)#tokenizes the words into tokens
    Ld[idx] = len(tokens)# doc_length stores number of words in each doc
    words=[]
    for word in tokens: #iterate over the words and add their respective stems to the storage.
        if len(word)>1:
            word=ps.stem(word)
            if word not in Stopwords:
                words.append(word)
    doc_words.append(words) # doc_words is a list of list for stemmed words for every doc
    counter = dict(Counter(words)) # count the frequency of each word in current doc
    dict_global.update(counter) # update the frequencies in global dictionary
    doc_word_count.append(counter) # doc_word_count list of dict containing word:freq for every doc
    #for w in words:
        #dict_global_test[w] = dict_global_test.get(w,0) + 1
    files_with_index[idx] = os.path.basename(fname)
    idx = idx + 1
    
unique_words_all = set(dict_global.keys())


# tf is a dict containing. 'cold':{'doc1':'freq1','doc2':'freq2',...}
# df is a dict containing. {'cold':0,'hot':0,...}
# df stores number of doc in which word has occured
tf = {x: {} for x in unique_words_all}
df = {x:0 for x in unique_words_all}


idx =0
for doc in doc_word_count:
    for i in doc.keys():
        df[i]=df[i]+1
        tf[i][idx]=doc[i]   
    idx=idx+1 
print("Finished")

# Ld is a list that contains number of token in each doc
Ltot = sum(Ld.values())
Ltot

doc_norm={}
idx=0
files_count = len(files_with_index)
for i in doc_word_count:
    l2=0
    for j in i.keys():
        l2+=((i[j]*math.log(files_count/df[j]))**2)
    doc_norm[idx]=(math.sqrt(l2))
    idx +=1


a_file = open("processed_docs/file_idx.pkl", "wb")
pickle.dump(files_with_index, a_file)
a_file.close()
a_file = open("processed_docs/unique_words_all.pkl", "wb")
pickle.dump(unique_words_all , a_file)
a_file.close()



import pickle
with open('processed_docs/posting_list.pkl','wb') as file:
    pickle.dump(tf,file)
    file.close()
    
with open('processed_docs/df.pkl','wb') as file:
    pickle.dump(df,file)
    file.close()
    
with open('processed_docs/doc_len.pkl','wb') as file:
    pickle.dump(Ld,file)
    file.close()
    
with open('processed_docs/doc_words.pkl','wb') as file:
    pickle.dump(doc_words,file)
    file.close()
    
with open('processed_docs/doc_norm.pkl','wb') as file:
    pickle.dump(doc_norm,file)
    file.close()


print("All pickle files generated")


now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)



