import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import  layers, models
os.chdir("C:\\Users\\kunjeshparekh\\Desktop\\KP\\IMS\\py\\project\\Hateful_Meme_DrivenData")
with open('train.jsonl') as f:
  train = [json.loads(jline) for jline in f.read().splitlines()]

with open('dev_seen.jsonl') as f:
  dev_seen = [json.loads(jline) for jline in f.read().splitlines()]

with open('dev_unseen.jsonl') as f:
  dev_unseen = [json.loads(jline) for jline in f.read().splitlines()]

with open('test_seen.jsonl') as f:
  test_seen = [json.loads(jline) for jline in f.read().splitlines()]

with open('test_unseen.jsonl') as f:
  test_unseen = [json.loads(jline) for jline in f.read().splitlines()]


train = pd.DataFrame(train) #has op
dev_seen=pd.DataFrame(dev_seen) #has op
dev_unseen = pd.DataFrame(dev_unseen) #has op
test_seen=pd.DataFrame(test_seen) #No op
test_unseen = pd.DataFrame(test_unseen) #No op

train.columns
dev_seen.columns
dev_unseen.columns

test_seen.columns
test_unseen.columns


train_1=pd.concat([train,dev_seen],axis=0)

X_train=train_1["text"]
y_train=train_1["label"]
X_test=test_seen["text"]

train_np=np.array(X_train)
train_sent=np.array(y_train)
test_np=np.array(X_test)

print(train_np[10:11])

import cv2
import glob

a=[cv2.resize(cv2.imread(img),(28,28)) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Hateful_Meme_DrivenData/memes/*.png")]
b=[str(img) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Hateful_Meme_DrivenData/memes/*.png")] #b has all the names of the imgs with path

cv2.imshow("img",np.array(a)[2,:,:,:])
cv2.waitKey(0) 
b[2]

img_index=[i[(i.find("\\")+1):(i.find(".png"))] for i in b]
len(img_index)
img_index_df=pd.DataFrame(img_index)
img_index_df.columns=(['id'])
img_index_df=img_index_df.astype('int')
img_index_df.info()
train_1.info()


new_df=pd.merge(img_index_df,train_1,how='left',on='id')
validation_df=pd.merge(img_index_df,dev_unseen,how='left',on='id')
test_seen_df=pd.merge(img_index_df,test_seen,how='left',on='id')


new_df.info()
validation_df.info()
test_seen_df.info()

train_2=new_df[new_df["img"].isna()==False]
validation_df_2=validation_df[validation_df["img"].isna()==False]
test_seen_df_2=test_seen_df[test_seen_df["img"].isna()==False]

train_2_img=np.array(a)[train_2.index,:,:,:] #Input for image model
train_2_img.shape

validation_2_img=np.array(a)[validation_df_2.index,:,:,:] #Input for image model
validation_2_img.shape

test_seen_img=np.array(a)[test_seen_df_2.index,:,:,:] #Input for image model
test_seen_img.shape


train_2=train_2.reset_index().iloc[:,1:]

validation_df_2=validation_df_2.reset_index().iloc[:,1:]
test_seen_df_2=test_seen_df_2.reset_index().iloc[:,1:]


#Random check to see if the index of the img and text data set are aligned
cv2.imshow("train_img",train_2_img[5358,:,:,:])
cv2.waitKey(0) 
train_2.iloc[5358,:]

cv2.imshow("validation_img",validation_2_img[438,:,:,:])
cv2.waitKey(0) 
validation_df_2.iloc[438,:]

cv2.imshow("test_seen_img",test_seen_img[438,:,:,:])
cv2.waitKey(0) 
test_seen_df_2.iloc[438,:]

words=train_2[train_2.label==1].text
words_good=train_2[train_2.label==0].text

word_count=pd.DataFrame(words.str.split(expand=True).stack().value_counts())
word_good_count=pd.DataFrame(words_good.str.split(expand=True).stack().value_counts())


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords') #1time
print(stopwords.words('english'))
#from nltk.tokenize import word_tokenize
#nltk.download('punkt') #1time
#nltk.word_tokenize(words)
#text_tokens = (words.apply(word_tokenize))
#tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]


top_bad_words=word_count[((word_count.index).isin (stopwords.words('english')))==False]
top_bad_words.columns=[("freq")]

top_good_words=word_good_count[((word_good_count.index).isin (stopwords.words('english')))==False]
top_good_words.columns=[("freq")]


import matplotlib.pyplot as plt

top_bad_words.iloc[:25,:].plot.barh()
plt.show()
top_bad_words.iloc[0:25,:]

top_good_words.iloc[:15,:].plot.barh()
plt.show()
top_good_words.iloc[0:15,:]

top_bad_words.iloc[:25,:].index

top_good_words.iloc[:35,:].index

#top_bad_words.iloc[:25,:].index[(top_bad_words.iloc[:25,:].index).isin(top_good_words.iloc[:35,:].index)]

top_bad_words.iloc[:35,:].index[((top_bad_words.iloc[:35,:].index).isin(top_good_words.iloc[:35,:].index))==False]

top_bad_final=['muslim', 'muslims', 'kill', 'jews','tranny', 'goat',
       'hate', 'islam', 'gay', 'trash', 'fucking','fuck'] #did manual cleanup on above and updated as left

train_2[str(top_bad_final[0])]=0
train_2[str(top_bad_final[0])][train_2['text'].str.contains(top_bad_final[0])]=1

for i in top_bad_final:
    train_2[str(i)]=0
    train_2[str(i)][train_2['text'].str.contains(i)]=1
    validation_df_2[str(i)]=0
    validation_df_2[str(i)][validation_df_2['text'].str.contains(i)]=1
    test_seen_df_2[str(i)]=0
    test_seen_df_2[str(i)][test_seen_df_2['text'].str.contains(i)]=1

train_2.label.value_counts()
train_2[train_2.label==1].iloc[:,4:].sum().sum()/3266
train_2[train_2.label==0].iloc[:,4:].sum().sum()/5734

(train_2[train_2.label==1].iloc[:,4:].sum(axis=1)).value_counts()
(train_2[train_2.label==0].iloc[:,4:].sum(axis=1)).value_counts()

(validation_df_2[validation_df_2.label==0].iloc[:,4:].sum(axis=1)).value_counts()

validation_df_2.label.value_counts()
validation_df_2[validation_df_2.label==1].iloc[:,4:].sum().sum()/200
validation_df_2[validation_df_2.label==0].iloc[:,4:].sum().sum()/340

test_seen_df_2.iloc[:,3:].sum(axis=1).value_counts()

#For Bigrams
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import unicodedata
import re
def basic_clean(text):
  """
  A simple function to clean up the data. All the words that
  are not designated as a stop word is then lemmatized after
  encoding and basic regex parsing are performed.
  """
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english') 
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]

words = basic_clean(''.join(str(train_2['text'].tolist())))

overall_bigrams=(pd.Series(nltk.ngrams(words, 2)).value_counts())[:10]

words_bad = basic_clean(''.join(str(train_2['text'][train_2.label==1].tolist())))

words_bad_bigrams=(pd.Series(nltk.ngrams(words_bad, 2)).value_counts())[:10]

