
import streamlit as st
import database as db
import facebook_loader as fc
from streamlit_extras import switch_page_button
import time 
from streamlit_option_menu import option_menu
import json
from IPython import get_ipython


# web modul
from selenium import webdriver
import requests
from bs4 import BeautifulSoup
from facebook_scraper import get_posts
import facebook_scraper
from facebook_scraper import _scraper, get_posts , get_reactors
from selenium.webdriver.common.by import By



st.set_page_config(page_title="main")


no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(no_sidebar_style, unsafe_allow_html=True)


st.subheader('AI Based Social Media Scaper')
# initializing database 
database = db.Database('social_crapper.db')
# initializaing user credential table
database.create_credential_table()
database.create_user_profile_table()
database.create_comment_table()
database.create_post_table()

     
# initializaing facebook_srapper
# getting the login in profile data
# fbook =  fc.FacebookLoader()


# st.markdown("<h1 align='center'>Social App</h1>", unsafe_allow_html=True)
credential_diction = dict()

def update_credential_table():
    email =  credential_diction['email']
    password = credential_diction['password']
    database.insert_credential(email, password)


# ___________________ LOGIN UI-----------------
# with st.sidebar: 

#  accepting user credential....
email_input = st.text_input('Username:', placeholder='Email/Contact', label_visibility='hidden')
password_input = st.text_input('Password:', placeholder='Password', label_visibility='hidden' , type='password')

# radio button for selecting a specific social media
social_choice = st.radio(
    "", ('Facebook', 'Twiiter', 'Intergram'), horizontal=True)

# update_mining = st.checkbox('Update Mining Data?', value=True) 

up_lay , login_lay = st.columns(2)

# Every form must have a submit button.
update_button = up_lay.button("Update Data")

# sign_up_button = sign_lay.button('Register')
is_update_complete = False
login_button = login_lay.button('Login')

# submit buttion clicked
if update_button:
    
    # validating user option ==> IF USER OPTION IS FACEBOOK
    if social_choice == 'Facebook':


        with st.spinner('Updating Corpus......'):  
            # print('facebook logging activate')
            from json import load

            try: 

                # temp = {'email': email_input, 'password': password_input }

                # with open('credentials.json', 'w') as file:
                #     json.dump(temp, file)

                filename = 'facebook_minner.ipynb'
                with open(filename) as fp:
                    nb = load(fp)

                for cell in nb['cells']:
                    
                    if cell['cell_type'] == 'code':
                        source = ''.join(line for line in cell['source'] if not line.startswith('%'))
                        exec(source, globals(), locals())
                    # if minin complete print success message ...
                    is_update_complete = True

                    # credential = {'email': email_input, 'password': password_input }
                    # with open('credentials.json', 'w') as file:
                    #     json.dump(credential, file)

                    update_credential_table()
                   
                st.write('COMPLETED')
            except Exception as ex:
               
                print(repr(ex))
                st.error('Error Connecting')

            else: 
                if is_update_complete:
                   
                    st.success('Mining Complete... ')
         
         



        # st.error('Mining Interupted..')

import sqlalchemy
import pandas as pd
# creating engine 


if login_button: 

    engine = sqlalchemy.create_engine('sqlite:///social_crapper.db')
    connection = engine.connect()

    login_detials = pd.read_sql_table('credentials_table', connection)

    if (login_detials['username'][0] == email_input) and (login_detials['password'][0] == password_input): 

        switch_page_button.switch_page('dashboard')
        st.success('validated')

    else:
        st.error('Invalid Credentials')




              

            


        
            
            
            


# db, an, rm, rem = st.tabs(['Dashboard', 'Analysis', 'Recommendation', 'R-Scrap'])

# with db: 

#     profile_button = st.button('Load Pofile')
    
#     if profile_button: 
#         try:
#             page = fbook.scrape_profile()
#             st.write(page.keys())
#             st.success('page loaded')
#         except:
#             st.error('unable to load data...')
            
        


