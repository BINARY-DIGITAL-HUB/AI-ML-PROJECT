# importing necessary module 
from selenium import webdriver
import requests
from bs4 import BeautifulSoup
from facebook_scraper import get_posts
import facebook_scraper
from facebook_scraper import _scraper, get_posts , get_reactors
from selenium.webdriver.common.by import By
import joblib
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import json
import string
import pandas as pd 
import numpy as np 
from nltk.corpus import stopwords
import string
import streamlit as st


class FacebookLoader():

    pt_stem = PorterStemmer()
    st = stopwords.words('english')

    def __init__(self,profile_json, comment_file, post_file):
        self.post_data = self.clean_raw_post(post_file)
        # st.write(self.post_data)
        self.comment_table = comment_file
        self.profile_dict = self.get_facebook_profile(profile_json)
        self.stopword = stopwords.words('english')

# ================================ POST ==============================================
    def clean_data(self, data):
        data = ''.join([d.lower() for d in data if not d in string.punctuation])
        return data

    def clean_raw_post(self, post_data): 
        
        post_data = post_data[['post_id','text', 'images_description']]
        
        post_data['text'] = post_data['text'].apply(lambda x: self.clean_data(x))
        post_data['images_description'] = post_data['images_description'].apply(lambda x: self.clean_data(x))
        return post_data

# cleaning text ===> removal of punctuation, stemming and stopword removal 
    def clean_dataset(self,dataset):

        token_word = [word.lower() for word in word_tokenize(dataset)]
        clean_word = [token for token in token_word if not token in self.stopword]
        clean_word = ' '.join([cl_word for cl_word in clean_word if not cl_word in string.punctuation])

        return clean_word
  
    # return sentiment from text
    def sentiment_analyzer(self, text, model):
        # clean data
        clean_data = self.clean_dataset(text)

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


    # load model ....
    # model = joblib.load('naive_bayes_sentiment_analyzer_model.jb')
        
    # LOAD USER FACEBOOK PROFILE
    def get_facebook_profile(self, profil_file):
        with open('profile.json', 'r') as file: 
            data = json.load(file)
        
        return data

    #  computing data vulnerability and recommending
    def personal_data_vulnerability_score(self,profile, datas):
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
            clean = self.clean_dataset(data).split()
            
        

            # clean user profile
            prof_data = self.clean_dataset(count_data).split()

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
    
    # function to calculate probability score
    def post_table_probability(self,_scores_text, _scores_image_text):
        self.post_data['text_score(%)'] = _scores_text
        self.post_data['images_description_score(%)'] = _scores_image_text
        final_score = [int((v[0] + v[1])*100) for v in zip(_scores_text , _scores_image_text)]
        self.post_data['final_score(%)'] = final_score

        return self.post_data
    

    def comment_table_probability(self,comment_text_prob, comment_reply_prob):
        print('deburg')
        # self.comment_table['comment_text(%)'] = comment_text_prob
        # self.comment_table['replies(%)'] = comment_reply_prob
        final_score = [int((v[0] + v[1])*100) for v in zip(comment_text_prob , comment_reply_prob)]
        self.comment_table['vunerability_score(%)'] = final_score

    
    def sentiment_score(self, texts, model):
        texts = [''.join(text) for text in texts]
        sentiments = []
        for text in texts: 
            if text == 'no content': 
                sentiment = ('Neutral', 0)
                sentiments.append(str(sentiment))
            elif text == 'no reply': 
                sentiment = ('Neutral', 0)
                sentiments.append(str(sentiment))
                
            else: 
                sentiment = self.sentiment_analyzer(text, model) 
                sentiments.append(str(sentiment))
            
        # hello... 
        print(sentiments)
        return sentiments
