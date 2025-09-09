#!/usr/bin/env python
# coding: utf-8

# # IMPORTING IMAGES......

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import PIL
import os
import cv2
import matplotlib.pyplot as plt
import pathlib


# In[2]:


cell_image_path = './/cell_images'

window_path = pathlib.Path(cell_image_path)
window_path


# In[3]:


dataset_path = list(window_path.glob('*/*.png'))
print ('Lenght of image dataset ======> ', len(dataset_path))

# accesssing individual folders 
parasitize_folder = list(window_path.glob('Parasitized/*.png'))
uninfected_folder = list(window_path.glob('Uninfected/*.png'))


# In[4]:


# create a diction of label and images set 

cell_image_dictionary = {
    'parasitized':parasitize_folder, 
    'uniffected':uninfected_folder
}

cell_image_label = {
    'parasitized':0,
    'uniffected':1
}


# In[5]:


# GETTING A SINGLE SAMPLE IMAGE FIRST

image = cv2.imread(str(cell_image_dictionary['parasitized'][7]))
print(image.shape)
plt.imshow(image)


# In[6]:


# USING THE DICTIONARY CREATED ABOVE TO LOAD THE ENTIRE IMAGE SET 

X = [] # actual images are stored in this variable....
y = [] # labels are stored in this variable...........

for title, paths in cell_image_dictionary.items():
    
    print('Loading Path =============> : ' , title)
    for path in paths:
        image = cv2.imread(str(path))
        X.append(image)
        y.append(cell_image_label[title])

print(f'{len(X)} Images Loaded completely .....................')


# # 2. DATA PREPROCESSING.............

# In[7]:


# COVERTING IMAGES INTO NUMPY ARRAY FOR EASY OPERATIONS


# In[8]:


X = np.array(X)
y = np.array(y)


# In[ ]:


# CONVERTING ALL IMAGES TO THE SAME DIMENSION....... (2D ARE NOT OF THE SAME SIZE)
#     NOW LETS COVERT ALL IMAGE TO SIZE 120 BY 120


# In[48]:


padded_X = []
for image in X:
    p_image=cv2.resize(image, (64,64))
    padded_X.append(p_image)

print('image resized complete..........')


# In[49]:


# coverting padded_X to a numpy array and printing first five image of padded_X (re-shaped X)
X_padded = np.array(list(padded_X))
[im.shape for im in X_padded[:5]]


# In[ ]:





# In[ ]:


# SCALLING IMAGE.... TO IMPROVE NORMALIZED DATA


# In[ ]:


# scaled_X = padded_X / 255


# In[50]:


# SPLITTING OF DATA INTO TRAINING AS TESTING SET.....

from sklearn import model_selection

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(X_padded, y, test_size=.25, random_state=0)

print('x train shape' , xtrain.shape)
print('y train shape' , ytrain.shape)
print('x test shape', xtest.shape)
print('y test shape', ytest.shape)


# # 1. DATA EXPLORATION (CELL IMAGES)

# In[ ]:





# In[11]:


para_size = len(parasitize_folder)
uninfec_size = len(uninfected_folder)

print('Parasitized Cells ======> ', para_size)
print ('Uninfected Ceslls =====>' , uninfec_size)


# In[12]:


# VISUALIZING DATA COUNT..... 

x_label = ['PARASITIZE CELL' , 'UN-AFFECTED CELL']
y_count = [para_size , uninfec_size]

plt.title('DATA SET COUNT')
plt.bar(x_label, y_count, color=['red', 'green'])
plt.bar


# In[13]:


# #  SHOW SAMPLE IMAGE FROM EACH CLASS 
labels = ['parasitized image' , 'un affected image']

#  getting index value of where label is 1 and 0 using it to access image in X numpy array 
affected , nonaffected = np.where(y == 0)[0] , np.where(y == 1)[0]
images = [X[affected[0]] , X[nonaffected[0]]]

plt.figure(figsize=(10,10))
for i in range(2):
    plt.subplot(1,2, i+1)
    plt.imshow(images[i])
    plt.xlabel(labels[i])

plt.show()


# # Malaria cell and Non Malaria Cell Sample

# In[14]:


# CHECKING FIRST 10 IMAGE DIMENSION 


# In[15]:


cell_samples  = X[:5]
print([im.shape for im in cell_samples])

print('\n COMMENT ..... ALL CELLS ARE 3 CHANELS IMAGES BUT OF DIFERENT SECOND DIMENSION (2D)... ')


# In[105]:


cell_samples[0]


# In[ ]:





# # 3. BUILDING THE CNN MODEL.....

# In[55]:


prediction_class = 1

cnn_model = keras.Sequential([
#     keras.layers.Conv2D(filters=64 , kernel_size=(3,3) , padding='same', activation='relu', input_shape=(64,64,3)),
    keras.layers.Conv2D(filters=32 , kernel_size=(3,3) , padding='same', activation='relu', input_shape=(64,64,3)),
    keras.layers.MaxPooling2D(),
    
    keras.layers.Conv2D(filters=64 , kernel_size=(3,3) , padding='same', activation='relu'),
    keras.layers.MaxPooling2D(), 
    
    keras.layers.Flatten(),
#     keras.layers.Dropout(.50),
    #     keras.layers.Dropout(.50),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(prediction_class, activation='sigmoid')
    
])



cnn_model.compile(
    loss = "binary_crossentropy",
    optimizer='adam',
    metrics=['accuracy']
)


# # Model Summary

# In[176]:


cnn_model.summary()
callback = tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1)


# In[ ]:





# # Training the model

# In[177]:


epoch = 7 # i try 2 epoch 15, 5 etc 
cnn_model.fit(xtrain, ytrain, batch_size=64, epochs=epoch, verbose=1 , callbacks=callback)


# In[178]:


cnn_model.evaluate(xtest, ytest, batch_size=64)


# In[ ]:





# # 4. EVALUATE THE MALARIA CELL CLASSIFICATION CCN MODEL

# In[179]:


# helper method to convert the continuose value of sigmoid output to discrete (0 and 1)

def continuose_discrete(predict):
    y_pred=[]
    for y in predict:
            if y>=0.5:
                y_pred.append(1)
            else: 
                y_pred.append(0)
                
    y_pred = np.array(y_pred)
    return y_pred


y_predict = cnn_model.predict(xtest)
y_pred = continuose_discrete(y_predict)


# # Classfication Report

# In[180]:


from sklearn import metrics

class_report = metrics.classification_report(ytest, y_pred)
confuse_matrix = metrics.confusion_matrix(ytest, y_pred)

print('================= CLASSIFICATION REPORT ======================')
print('==============================================================\n')
print(class_report)
print('==============================================================')


# In[ ]:





# # 6. CONFUSION MATRIC FOR RESULT VISUALIAZATION

# In[181]:


import seaborn as sns
import pandas as pd 

cm_pd = pd.DataFrame(confuse_matrix)
data = np.array([[2,3],[4,5]])

label= ["Malaria Cell" , "Non Malaria Cell"]
plt.figure(figsize=(8,6))
sns.heatmap(cm_pd, annot=True , fmt='d', xticklabels=label , yticklabels=label)
sns.set(font_scale=2)


# In[175]:





# In[ ]:




