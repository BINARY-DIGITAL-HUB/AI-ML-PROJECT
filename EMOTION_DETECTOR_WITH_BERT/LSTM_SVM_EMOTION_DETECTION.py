#!/usr/bin/env python
# coding: utf-8

# # Importing stweet dataset 

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import pandas as pd 

train_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/DATASET/EMOTION_DATA/archive/train.txt' ,names=['text', 'sentiment'], delimiter=';')
test_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/DATASET/EMOTION_DATA/archive/test.txt' ,names=['text',  'sentiment'], delimiter=';')
validate_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/DATASET/EMOTION_DATA/archive/val.txt' ,names=['text' , 'sentiment'], delimiter=';')


# In[ ]:


dataset = pd.concat([train_df, test_df, validate_df], axis=0)
dataset


# # Exploring Dataset (Data Exploration)

# In[ ]:


dataset.shape


# In[ ]:


dataset.columns 


# In[ ]:


dataset.text


# In[ ]:


dataset.sentiment.unique()


# In[ ]:


import matplotlib.pyplot as plt
dataset.sentiment.value_counts().plot(kind='bar' , xlabel='Sentiment' , ylabel='Data Point')
plt.title('Class Count Value Statistics')


# In[ ]:


dataset.sentiment.value_counts().plot(kind='pie', autopct = '%0.2f%%' , figsize=(10, 8) )


# In[ ]:





# # preprocessing data

# In[ ]:


pip install text_hammer


# In[ ]:


import text_hammer as th
from tqdm import tqdm_notebook

def text_preprocessing(df, col_name): 
    column = col_name
    df[column] = df[column].progress_apply(lambda x:str(x).lower())
    df[column] = df[column].progress_apply(lambda x:th.cont_exp(x))
    df[column] = df[column].progress_apply(lambda x:th.remove_emails(x))
    df[column] = df[column].progress_apply(lambda x:th.remove_html_tags(x))
    df[column] = df[column].progress_apply(lambda x:th.remove_special_chars(x))
    df[column] = df[column].progress_apply(lambda x:th.remove_accented_chars(x))
    
    return df

text_preprocessing(dataset, 'text')


# In[ ]:


def return_mx(text): 
    return len(text.split())

dataset['word_len'] =  dataset.text.progress_apply(lambda x: return_mx(x))


# In[ ]:


dataset.word_len.max()


# In[ ]:


dataset['sentiment'] = dataset.sentiment.astype('category')


# In[ ]:


sent_code = {'anger':0 , 'fear':1, 'joy':2, 'love':3, 'sadness':4, 'surprise':5}
sent_code


# In[ ]:


dataset.info()


# In[ ]:


dataset.sentiment


# In[ ]:


dataset['sentiment'] = dataset.sentiment.cat.codes


# In[ ]:


total =0 
for row in dataset.text.values:
    total += len(row.split())

print('total words is :', total)


# # further deep learning processing (LSTM)
# 
# ---
# 
# 

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import model_selection


# In[ ]:


# splitting dataset 

X_train, X_test, y_train, y_test  = \
model_selection.train_test_split(dataset['text'] , dataset['sentiment'], test_size=0.2)


# In[ ]:


token = Tokenizer(num_words=30000 , oov_token='<OOV>')
token.fit_on_texts(X_train)
XtrainSeq = token.texts_to_sequences(X_train)
XtestSeq = token.texts_to_sequences(X_test)


# In[ ]:


pad_len = 70
train_pad = pad_sequences(XtrainSeq, maxlen=pad_len, padding='post')
test_pad = pad_sequences(XtestSeq, maxlen=pad_len,  padding='post')


# In[ ]:


train_pad


# In[ ]:


vocab_size = 30000
embedding_dim = 100
pad_length =70

lstm_model = tf.keras.Sequential([
    
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=pad_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.1, recurrent_dropout=0.1)),
    # tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(6, activation='sigmoid')
    
])


lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )
lstm_model.summary()


# In[ ]:


lstm_model.fit(train_pad, y_train , epochs=10, validation_data=(test_pad, y_test), verbose=1)


# In[ ]:


from sklearn.metrics import classification_report

predict = lstm_model.predict(test_pad)
predict


# In[ ]:


predict[0]


# In[ ]:


# saving the biderectinal lstm model
from keras import models 

# lstm_model.save('/content/drive/MyDrive/Colab Notebooks/MODEL/bi_lstm_model.h5')
lstm_model =  models.load_model('/content/drive/MyDrive/Colab Notebooks/MODEL/bi_lstm_model.h5')


# In[ ]:


import numpy as np
predictions = []
for pred in predict: 
  predictions.append(np.argmax(pred))
predictions[:5]


# In[ ]:


from sklearn import metrics
report = metrics.classification_report(y_test, predictions)
print(report)


# In[ ]:


from sklearn import metrics
cm = metrics.confusion_matrix(y_test, predictions)
cm


# In[ ]:


import matplotlib.pyplot as plt
conf_fig = metrics.ConfusionMatrixDisplay(confusion_matrix=cm , display_labels=['anger', 'fear', 'joy', 'love', 'sadness', 'suprise'])
conf_fig.plot()
plt.show()


# In[ ]:


X_train


# # SUPPORT VECTOR MACHIN MODEL

# In[ ]:


# SVM MODEL
#  vectorizaing.. 
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
fit_xtrain  = cv.fit_transform(X_train)
fit_xtest  = cv.transform(X_test)
# cv.fit_transform()


# In[ ]:


fit_xtrain.shape


# In[ ]:


from sklearn import svm

sv = svm.SVC(C=0.1, gamma = 0.1, kernel='linear')
svm_model = sv.fit(fit_xtrain, y_train)


# In[ ]:


svm_model.score(fit_xtest, y_test)


# In[ ]:


# saving machine learning model
import joblib 

joblib.dump(svm_model , '/content/drive/MyDrive/Colab Notebooks/MODEL/svm_model.pk')


# In[ ]:


# Import the necessary libraries
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for the SVM model
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'],'gamma':[0.1,1]}

# Create the SVM model
svc = SVC()

# Create the grid search object
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')

# Fit the grid search object to the data
grid_search.fit(fit_xtrain, y_train)

# Print the best parameters and the best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# In[ ]:


from sklearn import metrics 
predict = svm_model.predict(fit_xtest)
report = metrics.classification_report(y_test, predict)
print(report)


# In[ ]:


import matplotlib.pyplot as plt
cm = metrics.confusion_matrix(y_test, predict)

conf_fig = metrics.ConfusionMatrixDisplay(confusion_matrix=cm , display_labels=['anger', 'fear', 'joy', 'love', 'sadness', 'suprise'])
conf_fig.plot()
plt.show()

