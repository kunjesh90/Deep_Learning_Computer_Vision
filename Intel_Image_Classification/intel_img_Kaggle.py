import zipfile
import pandas as pd
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import  layers, models

#image
   
#Read all the imgs in 3D numpy array 
#1.Buildings
a=[cv2.resize(cv2.imread(img),(150,150)) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_train/seg_train/buildings/*.jpg")]
b=[str(img) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_train/seg_train/buildings/*.jpg")] #b has all the names of the imgs with path

building = np.concatenate([arr[np.newaxis] for arr in a])
print(np.array(building).shape) #(2191, 150, 150, 3))

cv2.imshow("image",building[1,:,:,:])
cv2.waitKey(0)

#2.forest
a=[cv2.resize(cv2.imread(img),(150,150)) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_train/seg_train/forest/*.jpg")]
b=[str(img) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_train/seg_train/forest/*.jpg")] #b has all the names of the imgs with path

forest = np.concatenate([arr[np.newaxis] for arr in a])
print(np.array(forest).shape) #(2271, 150, 150, 3)

cv2.imshow("image",forest[1,:,:,:])
cv2.waitKey(0)

#3.glacier
a=[cv2.resize(cv2.imread(img),(150,150)) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_train/seg_train/glacier/*.jpg")]
b=[str(img) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_train/seg_train/glacier/*.jpg")] #b has all the names of the imgs with path

glacier = np.concatenate([arr[np.newaxis] for arr in a])
print(np.array(glacier).shape) #(2404, 150, 150, 3)
cv2.imshow("image",glacier[1,:,:,:])
cv2.waitKey(0) 

#4.mountain
a=[cv2.resize(cv2.imread(img),(150,150)) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_train/seg_train/mountain/*.jpg")]
b=[str(img) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_train/seg_train/mountain/*.jpg")] #b has all the names of the imgs with path

mountain = np.concatenate([arr[np.newaxis] for arr in a])
print(np.array(mountain).shape) #(2512, 150, 150, 3)
cv2.imshow("image",mountain[1,:,:,:])
cv2.waitKey(0) 

#5.sea
a=[cv2.resize(cv2.imread(img),(150,150)) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_train/seg_train/sea/*.jpg")]
b=[str(img) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_train/seg_train/sea/*.jpg")] #b has all the names of the imgs with path

sea = np.concatenate([arr[np.newaxis] for arr in a])
print(np.array(sea).shape) #(2274, 150, 150, 3)
cv2.imshow("image",sea[1,:,:,:])
cv2.waitKey(0) 

#6.street
a=[cv2.resize(cv2.imread(img),(150,150)) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_train/seg_train/street/*.jpg")]
b=[str(img) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_train/seg_train/street/*.jpg")] #b has all the names of the imgs with path

street = np.concatenate([arr[np.newaxis] for arr in a])
print(np.array(street).shape) #(2382, 150, 150, 3)
cv2.imshow("image",street[1,:,:,:])
cv2.waitKey(0) 
 

#Plot Images in a Grid 5X5 images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(building[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    #plt.xlabel(class_names[train_labels[i][0]])
plt.show()

class_names = ['Building', 'forest', 'glacier', 'mountain', 'sea','street']

train=np.vstack((building,forest,glacier,mountain,sea,street))
train.shape #(14034, 150, 150, 3)
len(building)+len(forest)+len(glacier)+len(mountain)+len(sea)+len(street)==train.shape[0]
(train[500,:,:,:]==building[500,:,:,:])

op_0=np.repeat(0,len(building))
op_0=op_0.reshape(op_0.shape[0],1)

op_1=np.repeat(1,len(forest))
op_1=op_1.reshape(op_1.shape[0],1)

op_2=np.repeat(2,len(glacier))
op_2=op_2.reshape(op_2.shape[0],1)

op_3=np.repeat(3,len(mountain))
op_3=op_3.reshape(op_3.shape[0],1)

op_4=np.repeat(4,len(sea))
op_4=op_4.reshape(op_4.shape[0],1)

op_5=np.repeat(5,len(street))
op_5=op_5.reshape(op_5.shape[0],1)

train_op=np.vstack((op_0,op_1,op_2,op_3,op_4,op_5))
len(train_op)==len(train)
#Train creation ends

#Test Set creation
#1.Buildings
a=[cv2.resize(cv2.imread(img),(150,150)) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_test/seg_test/buildings/*.jpg")]
b=[str(img) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_test/seg_test/buildings/*.jpg")] #b has all the names of the imgs with path

building = np.concatenate([arr[np.newaxis] for arr in a])
print(np.array(building).shape) #(437, 150, 150, 3))

cv2.imshow("image",building[1,:,:,:])
cv2.waitKey(0)

#2.forest
a=[cv2.resize(cv2.imread(img),(150,150)) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_test/seg_test/forest/*.jpg")]
b=[str(img) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_test/seg_test/forest/*.jpg")] #b has all the names of the imgs with path

forest = np.concatenate([arr[np.newaxis] for arr in a])
print(np.array(forest).shape) #(474, 150, 150, 3)

cv2.imshow("image",forest[1,:,:,:])
cv2.waitKey(0)

#3.glacier
a=[cv2.resize(cv2.imread(img),(150,150)) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_test/seg_test/glacier/*.jpg")]
b=[str(img) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_test/seg_test/glacier/*.jpg")] #b has all the names of the imgs with path

glacier = np.concatenate([arr[np.newaxis] for arr in a])
print(np.array(glacier).shape) #(553, 150, 150, 3)
cv2.imshow("image",glacier[1,:,:,:])
cv2.waitKey(0) 

#4.mountain
a=[cv2.resize(cv2.imread(img),(150,150)) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_test/seg_test/mountain/*.jpg")]
b=[str(img) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_test/seg_test/mountain/*.jpg")] #b has all the names of the imgs with path

mountain = np.concatenate([arr[np.newaxis] for arr in a])
print(np.array(mountain).shape) #(525, 150, 150, 3)
cv2.imshow("image",mountain[1,:,:,:])
cv2.waitKey(0) 

#5.sea
a=[cv2.resize(cv2.imread(img),(150,150)) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_test/seg_test/sea/*.jpg")]
b=[str(img) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_test/seg_test/sea/*.jpg")] #b has all the names of the imgs with path

sea = np.concatenate([arr[np.newaxis] for arr in a])
print(np.array(sea).shape) #(510, 150, 150, 3)
cv2.imshow("image",sea[1,:,:,:])
cv2.waitKey(0) 

#6.street
a=[cv2.resize(cv2.imread(img),(150,150)) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_test/seg_test/street/*.jpg")]
b=[str(img) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_test/seg_test/street/*.jpg")] #b has all the names of the imgs with path

street = np.concatenate([arr[np.newaxis] for arr in a])
print(np.array(street).shape) #(501, 150, 150, 3)
cv2.imshow("image",street[5,:,:,:])
cv2.waitKey(0) 

test=np.vstack((building,forest,glacier,mountain,sea,street))
test.shape #(3000, 150, 150, 3)
len(building)+len(forest)+len(glacier)+len(mountain)+len(sea)+len(street)==test.shape[0]
(test[50,:,:,:]==building[50,:,:,:])

op_0=np.repeat(0,len(building))
op_0=op_0.reshape(op_0.shape[0],1)

op_1=np.repeat(1,len(forest))
op_1=op_1.reshape(op_1.shape[0],1)

op_2=np.repeat(2,len(glacier))
op_2=op_2.reshape(op_2.shape[0],1)

op_3=np.repeat(3,len(mountain))
op_3=op_3.reshape(op_3.shape[0],1)

op_4=np.repeat(4,len(sea))
op_4=op_4.reshape(op_4.shape[0],1)

op_5=np.repeat(5,len(street))
op_5=op_5.reshape(op_5.shape[0],1)

test_op=np.vstack((op_0,op_1,op_2,op_3,op_4,op_5))
len(test_op)==len(test)
#test creation ends

#CNN
model = models.Sequential()
model.add(layers.Conv2D(150, (3, 3), activation='relu', input_shape=(150, 150, 3))) #shape of img is here 150,150,3 hence given the hyper parameters
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6)) #6 classes forr classification hence dense 6
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train, train_op, epochs=5, 
                    validation_data=(test, test_op))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test,  test_op, verbose=2)
print(test_acc)


probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
pred=probability_model.predict(test)
pred.shape
ans=(np.argmax(pred,axis=1))
(pd.DataFrame(ans).iloc[:,0]).value_counts()

#Final pred set

a=[cv2.resize(cv2.imread(img),(150,150)) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_pred/seg_pred/*.jpg")]
b=[str(img) for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_pred/seg_pred/*.jpg")] #b has all the names of the imgs with path

predset = np.concatenate([arr[np.newaxis] for arr in a])
print(np.array(predset).shape) #(7301, 150, 150, 3)
cv2.imshow("image",predset[1,:,:,:])
cv2.waitKey(0) 

predset_f=np.array(predset)

probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
pred=probability_model.predict(predset_f)
pred.shape
ans_final=(np.argmax(pred,axis=1))
(pd.DataFrame(ans).iloc[:,0]).value_counts()
final_pred=pd.concat([pd.DataFrame(b),pd.DataFrame(ans_final)],axis=1)
final_pred.head()
final_pred.to_csv("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/final_ans_pred.csv",index=False)

'''
with zipfile.ZipFile('C:\\Users\\kunjeshparekh\\Desktop\\KP\\IMS\\py\\project\\Intel_Image_Classification_Kaggle\\archive.zip', 'r') as zfile:
    data = zfile.read('seg_train/seg_train/buildings/1058.jpg')
    img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)

cv2.imshow("image",img)
cv2.waitKey(0) #0 is infinite delay , rest

cnt=0
directory=np.zeros((150,150,3))
directory=np.reshape(directory,(1,150,150,3))
for img in glob.glob("C:/Users/kunjeshparekh/Desktop/KP/IMS/py/project/Intel_Image_Classification_Kaggle/seg_train/seg_train/buildings/*.jpg"):
    cv_img = cv2.imread(img)
    new_img=np.reshape(cv_img,(1,150,150,3))
    directory = np.append(new_img , directory , axis = 0)
    print(cnt)
    cnt+=1
    #df[cnt]=cv_img
    #cnt+=1
    cv_img1=np.stack(cv_img,axis=0)
    #pd.concat([df,pd.DataFrame(cv_img)],axis=0)
 #   df=np.stack((df,cv_img),axis=0)
#    np.vstack()
    cnt+=1

directory.shape
'''
