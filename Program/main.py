# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tokenise as cln
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import time


 

data=pd.DataFrame.from_csv('labeledtrainData.tsv',sep='\t')

data.index=np.array(range(len(data)))

train_data=data.loc[:1000]
#train_data=data;
m=len(train_data);
test_data=data.loc[1500:1600];
n=len(test_data)
#test_data=pd.DataFrame.from_csv('testData.tsv',sep='\t')
test_data.index=np.array(range(len(test_data))) 


print('CLEANING TRAIN DATA....')
t=time.time()
i=1;
clean_train_review=[cln.clean(rev,m) for rev in train_data.review] 
print('\nTrain Data Has been cleaned in '+str(time.time()-t)+' s')

print('EXAMPLE OF CLEAN REVIEW BY ALGORITHM=\n')
print('GIVEN TRAIN REVIEW----------  \n'+str(train_data.review.loc[1]))
print('\nCLEAN REVIEW --------------\n'+str(clean_train_review[1]))

vectorizer=CountVectorizer(ngram_range=(1,3),analyzer='word',max_features=8000)

print('\nCreating train features..........')
t=time.time();
train_data_features=vectorizer.fit_transform(clean_train_review).toarray()
print('\nTraining  features created in '+str(time.time()-t)+' s')

vocab=vectorizer.get_feature_names()
train_target=np.array(train_data.sentiment).astype(int)

print('\nTRAINING DATA.............')
t=time.time();
clf=MultinomialNB()
classes=[0,1]
clf=clf.partial_fit(train_data_features,train_target,classes)
print('TRAINING COMPLETED IN '+str(time.time()-t)+' s')

pred=clf.predict(train_data_features)

print()
print('\nTRAIN ACCURACY='+str(np.mean(pred==train_data.sentiment)*100));

print('\nGETTING AND CLEANING TEST DATA NOW.............')
t=time.time();
clean_test_review=[cln.clean(rev,n) for rev in test_data.review] 
print('\nTEST DATA CLEANED IN '+str(time.time()-t)+' s')

test_data_features=vectorizer.transform(clean_test_review).toarray()

print("\nTESTING AVAILABLE TEST DATA NOW");
pred=clf.predict(test_data_features)

print()
print('Test ACCURACY='+str(np.mean(pred==test_data.sentiment)*100));

feedback_pos= np.sum((pred==1)&(test_data.sentiment==1))/np.sum(test_data.sentiment==1)
feedback_neg= np.sum((pred==0)&(test_data.sentiment==0))/np.sum(test_data.sentiment==0)

ch=1;

wrong_count=0
feed_train=[]
feed_target=[]
while ch!=0:
    test_string=input('ENTER UR MOVIE REVIEW TO TEST SENTIMENT............\n')
    cln_string=[cln.clean(test_string,0)]
    
    print('Your cleaned review is ='+str(cln_string))
    my_data=vectorizer.transform(cln_string).toarray()
    pred=clf.predict(my_data)
    if pred==1:
        print('YOUR REVIEW IS POSITIVE !!')
    else:
        print('YOUR REVIEW IS NEGATIVE !!')
#    feed=int(input('IS THAT CORRECT? PLEASE ENTER YOUR FEEDBACK REVIEW(1 for pos  0 for neg )'))
#    if feed!=pred:
#        feed_train.append(str(cln_string))
#        feed_target.append(feed)  
#        wrong_count=wrong_count+1
#        print('WRONG COUNT='+str(wrong_count))
#    if wrong_count==5:
#        X=vectorizer.transform(feed_train).toarray()
#        Y=np.array(feed_target)
#        print('fitting feedback data')
#        clf.partial_fit(X,Y)
#        feed_train=[]
#        feed_target=[]
#        wrong_count=0
    ch=(input('TO EXIT PRESS 0 \n'))
    if ch!='0':
        ch=1
    else:
        ch=0
        
    
#predictions=np.array([test_data.index,pred]).T
#df=pd.DataFrame(predictions,columns=['id','sentiment'])
#df.to_csv("Predictions.csv",index=False)

    