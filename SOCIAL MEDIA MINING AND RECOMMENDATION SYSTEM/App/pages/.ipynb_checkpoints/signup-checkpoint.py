import streamlit as st 
from streamlit_extras import switch_page_button
import streamlit as st

import re



class Validator:
    def __init__(self):
        pass

    def is_valid_email(self, email):
        pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        return re.match(pattern, email) is not None

    def is_valid_phone(self, phone):
        pattern = r'^\+?\d{10,15}$'
        return re.match(pattern, phone) is not None

    def is_valid_number(self, number):
        pattern = r'^[0-9]+$'
        return re.match(pattern, number) is not None


# with open('style.css') as css: 
#     st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)


no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
# st.markdown(no_sidebar_style, unsafe_allow_html=True)
# ==============  DATABSE CONNECTIVITIES ======================
# st.set_page_config(initial_sidebar_state="collapsed")





# ========================== END OF DATABASE CONNECTION ===============================

# defining global parameter
login_status = False


# app title
st.title('Social Media Scapper')


# widgets
st.header('Register Account')


# layout definations (three columns)
status_l1, statusl_l2 = st.columns([2,1])
name1, name2 = st.columns([2,1])
email1, email2 = st.columns([2,1])
password1, password2 = st.columns([2,1])
pass1, pass2 = st.columns([2,1])
sex_panel , age_panel = st.columns([2,1])




fname = name1.text_input("First Name")
lname = name2.text_input("Last Name")
email = email1.text_input("Email" )
# booking_id = email2.text_input('Booking ID')
booking_id = ' '
password = pass1.text_input('Password', type='password')
password2 = pass2.text_input('Confirm Password', type='password')

sex = sex_panel.selectbox('Select Gender' , ('Male', 'Female'))
age = age_panel.text_input("Enter Age")


col_one , col_two = st.columns([1,1])

back_button = col_two.button('Back')
if back_button: 
    switch_page_button.switch_page("main")



regitem  = [fname, lname, email, password, password2]

if col_one.button('Submit'):
    # validate data suply 
    pass

# col_two.checkbox('Remember Me?', value=True)

