#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install datasets


# In[ ]:


pip install transformers


# # Downloading Hugging Face Dataset 

# In[ ]:


# LOADING HUGGIN FACE TWEET-EMOTION DATA 
from datasets import load_dataset

dataset = load_dataset("tweet_eval", "emotion")


# In[ ]:


text


# In[ ]:


dataset


# In[ ]:


keys = dataset.keys()
# print(keys)
dataset['train'].column_names


# In[ ]:


dataset['train']['text']


# In[ ]:


id = dataset['train'].flatten().features['label'].names
print(id)


# In[ ]:


dataset['train'].flatten().features['label']


# LOADING DATASET.. 

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


import pandas as pd

tr_df = pd.read_csv('EMOTION_DATA/train.txt', sep=';', header=None, names=['text', 'label'])
ts_df = pd.read_csv('EMOTION_DATA/test.txt', sep=';', header=None, names=['text', 'label'])
vl_df = pd.read_csv('EMOTION_DATA/val.txt', sep=';', header=None, names=['text', 'label'])


# In[4]:


import pandas as pd 
data_full = pd.concat([tr_df, ts_df, vl_df], axis=0)


# In[5]:


data_full


# In[6]:


pip install text_hammer


# # Exploring Emotion Dataset

# In[8]:


dataset = data_full.copy()
rows , colums = dataset.shape

print(f'Number of Rows is  = {rows}')
print(f'Number of Columns is = {colums}')


# In[9]:


dataset.columns


# In[10]:


dataset.text[:10]


# In[ ]:


import matplotlib.pyplot as plt
dataset.label.value_counts().plot(kind='bar' , xlabel='Sentiment' , ylabel='Data Point')
plt.title('Class Count Value Statistics')


# In[14]:


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

text_preprocessing(data_full, 'text')


# In[ ]:


data_full


# In[15]:


df_cleaned = data_full.copy()


# In[16]:


df_cleaned['num_words'] = df_cleaned['text'].progress_apply(lambda x:len(x.split()))


# In[17]:


df_cleaned.info()


# In[18]:


df_cleaned['label'] = df_cleaned.label.astype('category')


# In[19]:


df_cleaned.info()


# In[20]:


df_cleaned.head()


# In[ ]:


df_cleaned.label


# In[21]:


df_cleaned['label'] = df_cleaned.label.cat.codes


# In[22]:


tags_dict = {'anger':0 , 'fear':1, 'joy':2, 'love':3, 'sadness':4, 'surprise':5}


# In[23]:


df_cleaned


# In[24]:


df_cleaned.num_words.max()


# In[25]:


from sklearn import model_selection

train_data , test_data = model_selection.train_test_split(df_cleaned, test_size=0.3, random_state=42 )


# In[27]:


train_data.shape


# In[28]:


test_data.shape


# In[ ]:


train_data.head()


# In[ ]:


pip install transformers


# In[ ]:


from transformers import AutoTokenizer
from transformers import TFBartModel
# from transformers import DistilBertTokenizer, DistilBertModel
from transformers import DistilBertTokenizer, TFDistilBertForMaskedLM
import tensorflow as tf


# In[ ]:


# checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'
# checkpoint = 'bert-base-cased'
# tokenizer = AutoTokenizer.from_pretrained(checkpoint )
# bert = TFBartModel.from_pretrained(checkpoint)
# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

# tensorlflow
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert = TFDistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")


# In[ ]:


# # SAVING THE MODEL
# tokenizer.save_pretrained('/content/drive/MyDrive/Colab Notebooks/MODEL/distilbert-tokenizer')
# bert.save_pretrained('/content/drive/MyDrive/Colab Notebooks/MODEL/distilbert_model')


# In[ ]:


# # SAVING THE MODEL
tokenizer.save_pretrained('/content/drive/MyDrive/Colab Notebooks/MODEL/bert-tokenizer')
bert.save_pretrained('/content/drive/MyDrive/Colab Notebooks/MODEL/bert_model')


# In[ ]:


# RELOADING MODEL 
tokenizer = AutoTokenizer.from_pretrained('/content/drive/MyDrive/Colab Notebooks/MODEL/distilbert-tokenizer')
bert = TFBartModel.from_pretrained('/content/drive/MyDrive/Colab Notebooks/MODEL/distilbert_model', output_attentions=True)


# In[ ]:


sample_text = [
    'this is a simple text', 
    'i love coding'
]
tokenizer(sample_text , padding=True)


# In[ ]:


wrd_lenght =df_cleaned.num_words.max()
wrd_lenght +=1 


# In[ ]:


# TOKENZin the dataset
# train_data.text.to_list()

x_train = tokenizer(
    text = train_data.text.to_list(), 
    add_special_tokens=True, 
    max_length= 70, 
    truncation=True, 
    padding=True, 
    return_tensors='tf', 
    return_token_type_ids=False, 
    return_attention_mask=True, 
    verbose=True
    )


x_test = tokenizer(
    text = test_data.text.to_list(), 
    add_special_tokens=True, 
    max_length= 70, 
    truncation=True, 
    padding=True, 
    return_tensors='tf', 
    return_token_type_ids=False, 
    return_attention_mask=True, 
    verbose=True
    )


# In[ ]:


# x_train
x_train['input_ids'][0]
x_train['attention_mask'][0]


# In[ ]:


import tensorflow as tf 
from tensorflow import keras
tf.config.experimental.list_physical_devices('GPU')


# In[ ]:


x_train.keys()


# In[ ]:


input_i = x_train['input_ids'][0]
sample_att = x_train['attention_mask'][0]


# In[ ]:


# weight = bert(input_i, attention_mask=sample_att)
# weight


# In[ ]:


max_len = 70

#  defining the input layer.... 
input_ids = keras.layers.Input(shape=(max_len,) , dtype= tf.int32, name='input_ids')
input_mask = keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')


# loading bert layer. 
embeddings = bert(input_ids,attention_mask = input_mask)[0]

# keras layer .. 
out1 = tf.keras.layers.GlobalMaxPool1D()(embeddings)
out2 = tf.keras.layers.Dense(128, activation='relu')(out1)
out3 = tf.keras.layers.Dropout(0.1)(out2)
out4 = tf.keras.layers.Dense(32, activation='relu')(out3)
final_output = tf.keras.layers.Dense(6, activation='sigmoid')(out4)

# defining the training model 
model =  tf.keras.Model(inputs=[input_ids, input_mask], outputs=final_output)
model.layers[2].trainable=True


# In[ ]:


model.layers


# In[ ]:


model.summary()


# In[ ]:


tf.keras.utils.plot_model(model, show_shapes=True)


# In[ ]:


from tensorflow.keras.optimizers import Adam
op = Adam(
    learning_rate = 5e-5, 
    epsilon = 1e-08, 
    decay = 0.01, 
    clipnorm=1.0   
)

loss = keras.losses.CategoricalCrossentropy(from_logits=True)
metric = keras.metrics.CategoricalAccuracy('balanced_accuracy')
model.compile(optimizer=op, 
             loss = loss,  
             metrics= metric)


# In[ ]:


x_train['attention_mask']


# In[ ]:


x_train['input_ids']


# In[ ]:


tf.config.experimental_run_functions_eagerly(True)
tf.config.run_functions_eagerly(True)


# 

# In[ ]:


from tensorflow.keras.utils import to_categorical


# In[ ]:


model.fit(
        x={'input_ids':x_train['input_ids'] , 'attention_mask':x_train['attention_mask']}, 
        y=  to_categorical(train_data.label), 
        validation_data=(  {'input_ids':x_test['input_ids'] , 'attention_mask':x_test['attention_mask']}, 
        to_categorical(test_data.label) ), epochs=10, batch_size=36
    )


# In[ ]:


raw_prediction = model.predict({'input_ids':x_test['input_ids'] , 'attention_mask':x_test['attention_mask']})


# In[ ]:


raw_prediction 


# In[ ]:


import numpy as np

test_data.label[:5]


# In[ ]:


prediction = [np.argmax(predict) for predict in raw_prediction]
prediction[:5]


# In[ ]:


from sklearn import metrics 
class_report = metrics.classification_report(test_data.label , prediction)
confu_matrix = metrics.confusion_matrix(test_data.label, prediction)
print(class_report)


# In[ ]:


import seaborn as sn 
import matplotlib.pyplot as plt
label = ['anger', 'fear', 'joy',  'love',   'sadness',  'surprise' ]
plt.figure(figsize=(12, 8))
sn.heatmap(confu_matrix, annot=True, yticklabels=label , xticklabels=label)


# In[30]:


train_data
test_data


# In[35]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

svm = SVC()
lstm = MLPClassifier()
bert = RandomForestClassifier()


models = [('SVM',svm), ('BERT', bert), ('LSTM', lstm) ]

# constructing the assemble model
ensemble_model =  VotingClassifier(
estimators=models , voting='hard')

# training the ensemple model
ensemble_model.fit(train_data['text'], train_data['label'])

