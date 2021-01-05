
import zipfile
import pandas as pd
import numpy as np
from PIL import Image
from io import StringIO

import cv2
import os
#Data Setting

archive = zipfile.ZipFile('C:\\Users\\kunjeshparekh\\Desktop\\KP\\IMS\\py\\project\\Apparel_Classification\\train.zip', 'r')

df=pd.DataFrame()
for i in range(1,60001):
    with archive.open('train/'+str(i)+'.png') as file:
        img = Image.open(file).convert('L') #for 3D .convert('LA')
    #img.show()
        print(i)
        a=(np.asarray(img))
    #print(img.size, img.mode, len(img.getdata()))
        b=np.array(a).reshape((28*28,1))/256
        df=pd.concat([df,pd.DataFrame(b.T)],axis=0)

df.info()

import os
os.getcwd()
os.chdir('C:\\Users\\kunjeshparekh\\Desktop\\KP\\IMS\\py\\project\\Apparel_Classification')
df.to_csv('train_pixel.csv',index=False) #To get away with reading the img dumping the pixels in csv flatten 28*28=784 columns
df.info()

archive = zipfile.ZipFile('C:\\Users\\kunjeshparekh\\Desktop\\KP\\IMS\\py\\project\\Apparel_Classification\\test.zip', 'r')

df_test=pd.DataFrame()
for i in range(60001,70001):
    with archive.open('test/'+str(i)+'.png') as file:
        img = Image.open(file).convert('L') #for 3D .convert('LA')
    #img.show()
        print(i)
        a=(np.asarray(img))
    #print(img.size, img.mode, len(img.getdata()))
        b=np.array(a).reshape((28*28,1))/255
        df_test=pd.concat([df_test,pd.DataFrame(b.T)],axis=0)
import os
os.getcwd()
os.chdir('C:\\Users\\kunjeshparekh\\Desktop\\KP\\IMS\\py\\project\\Apparel_Classification')
df_test.to_csv('test_pixel.csv',index=False)
df_test.info()
train=pd.read_csv(archive.open('train.csv')) 

#Classification Process
import zipfile
import os
import pandas as pd
os.getcwd()
os.chdir('C:\\Users\\kunjeshparekh\\Desktop\\KP\\IMS\\py\\project\\Apparel_Classification')
archive = zipfile.ZipFile('C:\\Users\\kunjeshparekh\\Desktop\\KP\\IMS\\py\\project\\Apparel_Classification\\train.zip', 'r')

train=pd.read_csv(archive.open('train.csv'))
train_img=pd.read_csv('train_pixel.csv')
test_img=pd.read_csv('test_pixel.csv')
test_img.columns


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_img,train["label"], test_size=0.05)
import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    
    keras.layers.Dense(256, activation='relu',input_shape=[len(X_train.columns)]),
    keras.layers.Dense(128, activation='relu',input_shape=[len(X_train.columns)]),
    keras.layers.Dense(64, activation='relu',input_shape=[len(X_train.columns)]),
    keras.layers.Dense(32, activation='relu',input_shape=[len(X_train.columns)]),
    #keras.layers.Dense(16, activation='relu',input_shape=[len(X_train.columns)]),
    #keras.layers.Dense(8, activation='relu',input_shape=[len(X_train.columns)]),
    keras.layers.Dense(10)
])

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.fit(X_train, y_train, epochs=75)

probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_img)

import numpy as np
op=np.argmax(predictions,axis=1)
(pd.DataFrame(op).iloc[:,0]).value_counts()
ans=pd.concat([pd.DataFrame([i for i in range(60001,70001)]),pd.DataFrame(op)],axis=1)
ans.columns = (['id','label'])
ans.to_csv("prediction.csv",index=False)
