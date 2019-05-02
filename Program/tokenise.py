# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
import re
import nltk
from bs4 import BeautifulSoup 
import nltk.stem
global i
i=1
from sys import stdout
def complete(m):
    global i
    if m==0:
        return
    print('\r'+str(int((i/m)*100))+'%',end='')
    stdout.flush()
    
    if i==m:
        i=1
    i=i+1



def clean(file,m):
    
    soup=BeautifulSoup(file,'lxml')
    #stop=set([word for word in (stopwords.words('english'))])
    s=nltk.stem.SnowballStemmer('english')    
    
    file=soup.get_text().lower()
    tokens=[re.match('[a-z]*',s.stem(word)).group() for word in nltk.word_tokenize(file)]
    
    k=nltk.pos_tag([word for word in tokens if word!='' ])

    #words=set([word for word,tag in k if (tag.startswith("NN") or tag.startswith("JJ") or tag.startswith("RB") or tag.startswith("VB") ) ])    
    
    string=" ";
    string=string.join(tokens)
    complete(m)

    
    return string
    
