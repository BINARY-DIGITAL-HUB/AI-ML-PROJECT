# %%
import streamlit as st 
import warnings
warnings.filterwarnings('ignore')
from selenium import webdriver
import requests
from bs4 import BeautifulSoup
from facebook_scraper import get_posts
import facebook_scraper
from facebook_scraper import _scraper, get_posts , get_reactors
from selenium.webdriver.common.by import By
import pandas as pd
import json
import sys

# %% [markdown]
# <h2 align='center'>AI Base Scrapping Tool for Facebook<h2>

# %%
# Enter mail: hamad1076663@gmail.com
# Password: Capstone12345

# %%


try:


# email = input('Enter Email: ')
# pass_code = input('Enter Passcode: ')

    email = sys.argv[1]
    pass_code = sys.argv[2]
    # connecting to browse
    browser = webdriver.Chrome("C://chromedriver")
    browser.get('https://web.facebook.com/')

    #summit credentials ju
    email_box = browser.find_element(by='id', value='email')
    password_box = browser.find_element(by='id', value='pass')
    login_button = browser.find_element(by='name', value='login')


    email_box.send_keys(email.strip())
    password_box.send_keys(pass_code.strip())

    login_flag = True
    print('login Successfull')
    st.success("code accepted")
# clicking button to login 
    login_button.click()
except Exception as e:
    login_flag = False
    print('login Fail')
    # print('envalid login {}'.format(e))

if login_flag: 
    credentials={'username':email, 'password':pass_code}
    with open('credentials.json', 'w') as file: 
        json.dump(credentials, file)
else: 
    credentials={'username':'', 'password':''}
    with open('credentials.json', 'w') as file: 
        json.dump(credentials, file)
    


# %%
# getting the login in profile data
def get_facebook_profile():
    
    browser.current_url
    # redirecting me to profile page
    profile_path = f'https://web.facebook.com/profile'
    browser.get(profile_path)
    
    return browser.current_url

profile = get_facebook_profile()



# using to extract profile name form link
def get_profile_name(profile):
    
    profile_name = profile.split('.com/')[-1].replace('/', '')
    
    return profile_name


user_profile_name = get_profile_name(profile)
user_profile_name
print('profile page loaded successfully')

# %%
# get the page bio data information
_scraper.login(email, pass_code)
page = facebook_scraper.get_profile(account=user_profile_name , images=True )
print('second security check loaded successful')

# %%
page.keys()

# %%
page.keys()

profile_keys = page.keys()
profile_keys = list(profile_keys)[1:-1]
print(profile_keys)

def clean_keys():
   clean_profile = {}
   for key in profile_keys: 
      temp_key = key.replace("\n", '').replace('Edit', '').strip()
      clean_profile[temp_key] =  page[key]

   return clean_profile

def add_mini_keys():
    data = clean_keys()
    mini_contact = data['Contact info'].replace('Edit', '').strip().split('\n')
    mini_key = mini_contact[1::2]
    mini_data = mini_contact[::2]
    for i in range(len(mini_key)): 
        data[mini_key[i]] = mini_data[i]

    return data

def clean_profile_information():
   page_data = add_mini_keys()
   page_data_keys = page_data.keys()

   clean_facebook_profile = {}
   for key in page_data_keys: 
      # print(key)
      val = page_data[key]
      if val != None:
         if key == 'Places lived': 
            val = page_data[key][0]['text']
            clean_facebook_profile[key.lower()] = val
            # print(key ,':----', val)

         elif key == 'Contact Info':
            for cont_key, cont_val in page_data[key].items():
               clean_facebook_profile[cont_key.lower()] = cont_val 
               # print(cont_key, ':--', cont_val)
         elif key == 'profile_picture':
            pass

         else:
            clean_facebook_profile[key.lower()] = val
            # print(key,':--', val)

   return clean_facebook_profile

# print(profile_keys['Contact info\nEdit'])
clean_profile_data = clean_profile_information()


# %%
add_mini_keys()
clean_profile_data

# %%
from PIL import Image
from io import BytesIO
import requests
import numpy as np
import json 
import cv2


def get_profile_key():
    profile_keys = page.keys()
    profile_keys = list(profile_keys)[1:]
    return profile_keys

def save_profile_info(page):
    profile_pix = page['profile_picture']
    img = Image.open(BytesIO(requests.get(profile_pix).content))
    print(type(img))
    img = np.array(img)
    cv2.imwrite('passport_image.jpg', img)
    json_info = clean_profile_information()
    print(json_info)
    # contact_data
    with open('profile.json', 'w') as file: 
            json.dump(json_info, file)
            print('profile dumb successful... ')
    

save_profile_info(page)

# %%
from facebook_scraper import get_posts


def load_posts(user_profile_name): 
    posts = []
    for post in get_posts(account=user_profile_name, extra_info=True, options={"comments":True, 'reactors':True}):
        posts.append(post)

    print('No Of Post : {}'.format(len(posts)))
    return posts

posts = load_posts(user_profile_name)


# %%

raw_data = ['post_id', 'text', 'time', 'images_description', 'reactions', 'post_url']

raw_data_diction = {
    'post_id':[],
    'text': [], 
    'time': [], 
    'images_description': [],
    'reactions': [],
    'post_url':[]
}

raw_comment_diction={
    'post_id':[], 
    'commenter_name':[],
    'comment_time':[],
    'comment_text':[], 
    'replies':[]
}

for post in posts[:]:
    # print('--------')
    for raw in raw_data: 
        # print(post[raw])
        if post[raw] != []:
            raw_data_diction[raw].append(post[raw])
        else:
            raw_data_diction[raw].append('')
    
    
#   init comment table with replies
    for comments in post['comments_full']: 
        raw_comment_diction['post_id'].append(post['post_id'])
        
        if comments['comment_text'] != []:
            raw_comment_diction['comment_text'].append(comments['comment_text'])
        else:
            raw_comment_diction['comment_text'].append('no comment')

        # if comments['images_description'] != []:
        #     raw_comment_diction['images_description'].append(comments['images_description'])
        # else:
        #     raw_comment_diction['images_description'].append('no comment')

        
        if comments['replies'] != []:
            raw_comment_diction['replies'].append(''.join(comments['replies']))
        else: 
            raw_comment_diction['replies'].append(''.join(["no reply"]))

        raw_comment_diction['commenter_name'].append(comments['commenter_name'])
        raw_comment_diction['comment_time'].append(comments['comment_time'])

raw_comment_diction

# %%
import pandas as pd

post_table = pd.DataFrame(raw_data_diction)
post_table.head()

# %%
comment_table = pd.DataFrame(raw_comment_diction)
comment_table.head()

# %% [markdown]
# Post Data 

# %%
import pandas as pd 
import numpy as np 
from nltk.corpus import stopwords
import string

# postdata = pd.read_csv('user_post_data.csv')
post_data = post_table[['post_id','text', 'images_description']]
post_data

st = stopwords.words('english')
post_data = post_data.replace([np.NAN, '[]', ''], 'no content')
punc = '!"#$%&\'()*,-/:;<=>?[\\]^_`{|}~'
def clean_data(data):
    data= ''.join([d.lower() for d in data if not d in punc])
    return data

post_data['text'] = post_data['text'].apply(lambda x: clean_data(x))
post_data['images_description'] = post_data['images_description'].apply(lambda x: clean_data(x))
post_data

# %%
post_data.to_csv('user_post_data.csv', index=False)
comment_table.to_csv('comment_table.csv.', index=False)

# %%
import pandas as pd 

post_data = pd.read_csv('user_post_data.csv')
comment_table = pd.read_csv('comment_table.csv.')

# %% [markdown]
# Saving Comment an Post

# %% [markdown]
# Sentiment analyzer and probability score

# %%
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string

pt_stem = PorterStemmer()
stop_word = stopwords.words('english')

# %%

from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import json
import string
import joblib

pt_stem = PorterStemmer()
st = stopwords.words('english')

# cleaning text ===> removal of punctuation, stemming and stopword removal 
def clean_dataset(dataset):

    token_word = [word.lower() for word in word_tokenize(dataset)]
    clean_word = [token for token in token_word if not token in st]
    clean_word = ' '.join([cl_word for cl_word in clean_word if not cl_word in string.punctuation])

    return clean_word

# LOADING USER FACEBOOK PROFILE... 
def get_facebook_profile():
    with open('profile.json', 'r') as file: 
        data = json.load(file)
    return data


profile_dict = get_facebook_profile()
# print(profile_dict)


def personal_data_vulnerability_score(profile, datas):
    prof_data = []
    for k,v in profile.items():
        prof_data.append(str(v).lower())
        # print(v)
        

    count_data = ' '.join(prof_data)
    # print(count_data)

    probability_score = []
    # clean comement.. and post
    for data in datas: 
        # print(data)
        clean = clean_dataset(data).split()
        
    

        # clean user profile
        prof_data = clean_dataset(count_data).split()

        prob_count = 0
        for com in clean: 
            # print(com)
            if com.lower() in prof_data: 
                prob_count += 1

        if len(prof_data) == 0: 
            
            probability_score.append(0)
             
        else: 
            probility_count = prob_count/len(prof_data)
            probability_score.append(probility_count)
    
    return probability_score

# load model ....
model = joblib.load('naive_bayes_sentiment_analyzer_model.jb')
def sentiment_analyzer(text, model):
    # clean data
    clean_data = clean_dataset(text)

    # loading tfid model .... 
    tfid = joblib.load('tfidf_sentiment_vocab.pk')

    # vectorizing
    word_vector = tfid.transform([clean_data])

    # load_model
    sentiment_model = model
    # make prediction
    predict = sentiment_model.predict(word_vector.toarray())

    sent_string = ''

    if predict[0] == 0:
        sent_string = 'Negative'
    else: 
        sent_string = 'Postive'


    return sent_string, predict[0]  


def sentiment_score(texts, model):
    texts = [''.join(text) for text in texts]
    sentiments = []
    for text in texts: 
        # print('MY TEXT : ', text=='no reply')
        
        if text == 'no reply':
            sentiment = ('Neutral', 0.5)
            sentiments.append(sentiment)
        else:
            sentiment = sentiment_analyzer(text, model) 
            sentiments.append(sentiment) 
            
        
    # hello... 
    print(sentiments)
    return sentiments


# vulnerability score
_scores_text  = personal_data_vulnerability_score(profile_dict, post_data['text'].values)
_scores_image_text = personal_data_vulnerability_score(profile_dict, post_data['images_description'])  



# function to calculate probability score
def post_table_probability(_scores_text, _scores_image_text):
    post_data['text_score(%)'] = _scores_text
    post_data['images_description_score(%)'] = _scores_image_text
    final_score = [int(v[0] + v[1]) for v in zip(_scores_text , _scores_image_text)]
    post_data['final_score(%)'] = final_score

post_table_probability(_scores_text, _scores_image_text)
post_data


# adding sentiment column 
# post_data['text_sent'] = _sentiment_text 
post_data

# %% [markdown]
# Comment Data

# %%
# _scores  = personal_data_vulnerability_score(profile_dict, com['text'].values)
comment_table = comment_table[['post_id', 'comment_text', 'replies']]

comment_text_prob = personal_data_vulnerability_score(profile_dict, comment_table['comment_text'].values)
comment_reply_prob = personal_data_vulnerability_score(profile_dict, comment_table['replies'].values)

print(comment_text_prob, comment_reply_prob)

def comment_table_probability(comment_text_prob, comment_reply_prob):
    print('deburg')
    comment_table['comment_text(%)'] = comment_text_prob
    comment_table['replies(%)'] = comment_reply_prob
    final_score = [int((v[0] + v[1])*100) for v in zip(comment_text_prob , comment_reply_prob)]
    comment_table['final_score(%)'] = final_score

# probability score
comment_table_probability(comment_text_prob, comment_reply_prob)

# sentiment
_sentiment_com_text  = sentiment_score(comment_table['comment_text'].values, model)
_sentiment_reply_text  = sentiment_score(comment_table['replies'].values, model)

# adding sentiment column 
comment_table['comment_sent'] = _sentiment_com_text 
comment_table['reply_sent'] = _sentiment_reply_text 

comment_table

# %%
post_data.to_csv('user_post_data.csv', index=False)
comment_table.to_csv('comment_table.csv', index=False)

# %%
import pandas as pd 

pt = pd.read_csv('comment_table.csv')
pt

# %%
recommendation_holder = []
for index in pt.index:
    # print(pt.iloc[index]['text'])
    score = pt.iloc[index]['final_score(%)']
    recommendation_holder.append(score)



# %% [markdown]
# DATABASE UPDATING...

# %%
import pandas as pd 
import sqlalchemy


# creating engine 
engine = sqlalchemy.create_engine('sqlite:///social_crapper.db')
connection = engine.connect()

credential_table  = pd.read_sql_table('credential_table', connection) 
credential_table

# %% [markdown]
# UPDATA CREDENTIAL DATA

# %%
import json
import pandas as pd

with open('credentials.json', 'r') as file: 
    credentials = json.load(file)

if credentials['username'] != "": 
    cre_table = pd.DataFrame(credentials, index=[0])
    # print(cre_table)

    cre_table.to_sql(
    name="credential_table", 
    con=connection, 
    if_exists="replace", 
    index=False
    )
    
    connection.commit()
    print("update successful 1")
else: 
    cre_table = pd.DataFrame(credentials, index=[0])
    # print(cre_table)

    cre_table.to_sql(
    name="credential_table", 
    con=connection, 
    if_exists="replace", 
    index=False
    )
    
    connection.commit()
    print("update successful 2")
    print("Credentials Invalid")


# %% [markdown]
# CREATING POST TABLE 

# %%
import pandas as pd 
com_table = pd.read_csv('user_post_data.csv')
com_table.to_sql(
    name="post_table", 
    con=connection, 
    if_exists="replace", 
    index=False
)
connection.commit()

# %% [markdown]
# CREATING COMMENT TABLE 

# %%
import pandas as pd 
com_table = pd.read_csv('comment_table.csv')
com_table.to_sql(
    name="comment_table", 
    con=connection, 
    if_exists="replace", 
    index=False
)
connection.commit()

# %% [markdown]
# UPDATING PROFILE DATA

# %%
import json
import pandas as pd

with open('profile.json', 'r') as file: 
    profil_j = json.load(file)

prof_table = pd.DataFrame(profil_j, index=[0])

prof_table.to_sql(
name="user_profile", 
con=connection, 
if_exists="replace", 
index=False
)

connection.commit()
print("update successful")

prof_table


# %%
# DISCONNECTING 
connection.close()


