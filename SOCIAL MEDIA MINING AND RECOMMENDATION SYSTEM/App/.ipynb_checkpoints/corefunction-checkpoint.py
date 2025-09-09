
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
import seaborn as sn
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib  import style
import joblib
import string 
from nltk.corpus import stopwords
import re 

stop_word = stopwords.words('english')

model = joblib.load('naive_bayes_sentiment_analyzer_model.jb')
    # clean data


# loading tfid model .... 
tfid = joblib.load('tfidf_sentiment_vocab.pk')

nlp_model = spacy.load('en_core_web_sm')



def get_entities(posts):

    post_entity_diction = {}
    for post in posts:
        # loading of spacy model
        
        # padding comment to model
        doc = nlp_model(post)

        # going throught the entity
        for ent in doc.ents: 
    

            if ent.label_ == 'DATE' or ent.label_ == 'CARDINAL': 
                pass
            else: 
                if post_entity_diction.get(ent.label_) != None: 
                    post_entity_diction[ent.label_].append(ent.text)
                else: 
                    post_entity_diction[ent.label_] = []
                    post_entity_diction[ent.label_].append(ent.text)

    return post_entity_diction

# post_entity = get_entities([post, post2])
# post_entity
# cleaning text ===> removal of punctuation, stemming and stopword removal 
def clean_dataset(dataset):

    token_word = [word.lower() for word in word_tokenize(dataset)]
    clean_word = [token for token in token_word if not token in stop_word]
    clean_word = ' '.join([cl_word for cl_word in clean_word if not cl_word in string.punctuation])

    return clean_word

# sentiment analyzer
# return sentiment from text
def sentiment_analyzer(text):

    clean_data = clean_dataset(text)
    
    # vectorizing
    word_vector = tfid.transform([clean_data])

    # load_model
    sentiment_model = model
    # make prediction
    predict = sentiment_model.predict(word_vector.toarray())

    sent_string = ''

    if predict[0] != 0:
        sent_string = 'Negative Sentiment'
    else: 
        sent_string = 'Postive Sentiment'


    return sent_string, predict[0]  

# getting contact and email pattern from post...
# getting contact and email pattern from post...
def contact_email_detector(posts):
    contact_diction = {'post_id':[], 'contacts':[] , 'emails':[],'contact_len':[], 'email_len':[]}
    counter = 0
    for post in posts: 
        counter += 1
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        uai_phone_pattern = r"\+971 [2355679]\d{1} \d{7}"


        matches = re.findall(uai_phone_pattern, post)
        contact_diction['contacts'].append(matches)
        contact_diction['contact_len'].append(len(matches))

        emails =  re.findall(email_pattern, post)
        contact_diction['emails'].append(emails)
        contact_diction['email_len'].append(len(emails))


        contact_diction['post_id'].append('Post {}'.format(counter))
        

    return contact_diction


# post1 = '+971 2 1234567 and +971 50 9876543 are valid  +971 8 5555555 UAE phone numbers, but not +971 1 5555555 or +971 55 1234567 and my email is hamad1076663@gmail.com.'
# post2 = 'the name of the boy is ahmed, adam and james and email contack  +971 8 5555555 +971 5 9876543   +971 2 1234567'
# contact_data = contact_email_detector([post1, post2])
# print(contact_data)

# READING COMMENT DATA 
def get_conmment_table(): 
    return pd.read_csv('comment_table.csv')


# google map data 
import pandas as pd 
import streamlit as st
import requests


def get_google_location(location): 
    location_corpus = {
        'place':[], 
        'name':[], 
        'lat':[], 
        'lon':[], 
        'importance':[]
    }
    # Use OpenStreetMap Nominatim API to get the latitude and longitude coordinates
    url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json"
    location_data = requests.get(url).json()
#     print(location_data[0].keys())
    if location_data != []: 
        
        print('possible location for  {} ==> {}'.format(location, len(location_data)))
        for location in location_data: 
            location_corpus['place'].append(location['place_id'])
            location_corpus['name'].append(location['display_name'])
            location_corpus['lat'].append(location['lat'])
            location_corpus['lon'].append(location['lon'])
            location_corpus['importance'].append(location['importance'])
    else: 
        st.warning('location not detected... ')
    # st.write(location_corpus)
    return location_corpus , len(location_data)

    
# data = get_google_location(input('Enter a location:'))