> Programming Language: Python 3.9.7

> Environment: Anaconda 4.11.0

### Libraries Needed:
numpy
pandas
glob
nltk
glob
re
os
sys
math
pathlib
collections
pickle

Steps to execute:

1. Make sure you install all the libraries mentioned above before running the assignment.
2.  Unzip 21111048-assignment1.zip in your local folder.
3. Open the terminal window in the same folder where you extracted 
4. Issue the command, "make run path={path to the input query file}", and wait for ~2/3 minutes
5.  The output of all the three IR Systems should be present in the "Output" directory in the format as 
        as specified in the question. Their names would be:
        1. BM25.csv        
        2. BRS.csv         
        3. TFIDF.csv

### NOTE

1. The entire preprocessing is already performed on the ~8500 text files and the output files
are saved in the folder "processed_docs". The corresponding code is contained in "q1.py". If you wish 
to recreate the processed documents, please use the command, "make setup". HOWEVER, THIS WILL TAKE
AROUND ~15 MINUTES.

2. The corpus is not included due to its huge size. If you want to do the preprocessing as mentioned above,
please download the corpus from "https://www.cse.iitk.ac.in/users/arnabb/ir/english/" and extract it in the
same folder as Makefile. At the end there should be a folder named "english-corpora" containing all
the raw text files.

3. Do not unzip the compressed files manually. Use the command, "make run path={path to the input query file}".
It should do all the needful.

4. Processed files from part 1 and output files can be found at: https://iitk-my.sharepoint.com/:f:/g/personal/pranshus21_iitk_ac_in/EvqhANfKZh5InUSK6pKpsDcBGFNkBE6Mb2pRrUs8e1XG9w?e=6Gu8Be
  
