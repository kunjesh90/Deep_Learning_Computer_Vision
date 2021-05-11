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

vocab_size = 10000
embedding_dim = 128
max_length = 1000
trunc_type='post'
oov_tok = "<OOV>"

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tkr = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tkr.fit_on_texts(train_np)
word_index = tkr.word_index
sequences = tkr.texts_to_sequences(train_np)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

testing_sequences = tkr.texts_to_sequences(test_np)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

np.max(padded) 

'''
#Only Word Embedding with NN
model_conv = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
#   tf.keras.layers.Flatten(),
#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(26, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_conv.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_conv.summary()

num_epochs = 12
model_conv.fit(padded, np.array(y_train), epochs=num_epochs, batch_size=128, validation_data=(padded, np.array((y_train))))
predictions_conv = model_conv.predict(testing_padded)
'''

#Method 1 Only Text
#for non Embedding + LSTM use below
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#    tf.keras.layers.GlobalAveragePooling1D(),
#    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
#    tf.keras.layers.Dense(72, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs =10
model.fit(padded, np.array(y_train), epochs=num_epochs, batch_size=128, validation_data=(padded, np.array((y_train))))
predictions = model.predict(testing_padded)
predictions.max()

df_pred_seen=pd.concat([test_seen.id,pd.DataFrame(predictions)],axis=1)
df_pred_seen['label']=np.where(df_pred_seen.iloc[:,1]<0.5,0,1)

df_pred_seen.columns=['id','proba','label']
df_pred_seen.to_csv("prediction_seen.csv",index=False)

'''
#For Embedding+GRU
model_gru = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
#    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
#    tf.keras.layers.Dense(72, activation='relu'),
#    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(26, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_gru.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_gru.summary()
'''

'''
#Method 2 Text+Images
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

'''
'''
img_model = models.Sequential()
img_model.add(layers.Conv2D(10, (3, 3), activation='relu', input_shape=(28, 28, 3))) #shape of img is here 150,150,3 hence given the hyper parameters 10 are the number of channels of the filters which should increase and dimention of l*b to reduce
img_model.add(layers.MaxPooling2D((2, 2)))
img_model.add(layers.Conv2D(32, (3, 3), activation='relu'))
img_model.add(layers.MaxPooling2D((2, 2)))
img_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
img_model.add(layers.Flatten())
img_model.add(layers.Dense(64, activation='relu'))
img_model.add(layers.Dense(32, activation='relu')) #6 classes forr classification hence dense 6
img_model.summary()

nlp_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#    tf.keras.layers.GlobalAveragePooling1D(),
#    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
#    tf.keras.layers.Dense(72, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(26, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu')])

nlp_model.summary()

combined = tf.keras.layers.concatenate([img_model.output, nlp_model.output])
combined=tf.keras.layers.Dense(10, activation="relu")(combined)
combined=tf.keras.layers.Dense(1, activation="sigmoid")(combined)

full_model = tf.keras.Model(inputs=[tf.keras.layers.Input(shape=(28, 28, 1)), tf.keras.layers.Input(shape=(28, 28, 1))], outputs=[combined])

print(full_model.summary())

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
'''
'''


# Image
input_1 = tf.keras.layers.Input(shape=(28, 28, 3))
conv2d_1 = tf.keras.layers.Conv2D(10, (3,3),
                                  activation=tf.keras.activations.relu)(input_1)
max2d_1 = tf.keras.layers.MaxPooling2D((3,3))(conv2d_1)
# Second conv layer :
conv2d_2 = tf.keras.layers.Conv2D(15, (3,3),activation=tf.keras.activations.relu)(max2d_1)
max2d_2 = tf.keras.layers.MaxPooling2D((3,3))(conv2d_2)
conv2d_3 = tf.keras.layers.Conv2D(5,(1,1),activation=tf.keras.activations.relu)(max2d_2)
# Flatten layer :
flatten = tf.keras.layers.Flatten()(conv2d_3)

vocab_size = 10000
embedding_dim = 42
max_length = 1000
trunc_type='post'
oov_tok = "<OOV>"

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tkr = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tkr.fit_on_texts(np.array(train_2.text))
word_index = tkr.word_index
sequences = tkr.texts_to_sequences(np.array(train_2.text))
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)
np.max(padded) #Input for text model

validation_sequences = tkr.texts_to_sequences(np.array(validation_df_2.text))
validation_padded = pad_sequences(validation_sequences,maxlen=max_length)

test_seen_sequences = tkr.texts_to_sequences(np.array(test_seen_df_2.text))
test_seen_padded = pad_sequences(test_seen_sequences,maxlen=max_length)

#

# Text
input_2 = tf.keras.layers.Input(shape=(1000,))

dense_2_1=tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length)(input_2)
dense_2_2=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(dense_2_1)

dense_2_3 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(dense_2_2)


# Concatenate
concat = tf.keras.layers.Concatenate()([flatten, dense_2_3])

n_classes = 1
# output layer
output = tf.keras.layers.Dense(units=n_classes,
                               activation=tf.keras.activations.sigmoid)(concat) #activations.softmax

full_model = tf.keras.Model(inputs=[input_1, input_2], outputs=[output])

print(full_model.summary())

full_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

num_epochs =8
full_model.fit([train_2_img,padded], np.array(train_2.label), epochs=num_epochs, batch_size=128)


import numpy as np

predictions_val = full_model.predict([np.array(validation_2_img, dtype='float'),validation_padded])


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(validation_df_2.label, predictions_val)
metrics.auc(fpr, tpr)

predictions_test_seen = full_model.predict([np.array(test_seen_img, dtype='float'),test_seen_padded])

df_pred_seen=pd.concat([test_seen.id,pd.DataFrame(predictions_test_seen)],axis=1)
df_pred_seen['label']=np.where(df_pred_seen.iloc[:,1]<0.5,0,1)

df_pred_seen.columns=['id','proba','label']
df_pred_seen.to_csv("prediction_seen_Concat.csv",index=False)


'''
# Text
'''

input_2 = tf.keras.layers.Input(shape=(1000,))

dense_2 = tf.keras.layers.Dense(5, activation=tf.keras.activations.relu)(input_2)

# Concatenate
concat = tf.keras.layers.Concatenate()([flatten, dense_2])

n_classes = 1
# output layer


output = tf.keras.layers.Dense(units=n_classes,
                               activation=tf.keras.activations.sigmoid)(concat) #activations.softmax

full_model = tf.keras.Model(inputs=[input_1, input_2], outputs=[output])

print(full_model.summary())

full_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
'''