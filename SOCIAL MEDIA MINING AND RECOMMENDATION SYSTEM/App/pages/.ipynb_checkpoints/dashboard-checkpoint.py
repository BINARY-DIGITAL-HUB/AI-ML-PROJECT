import streamlit as st
from PIL import Image
import json
from streamlit_extras import switch_page_button
import pandas as pd
import facebook_loader as fl
import joblib
import seaborn as sn
import matplotlib.pyplot as plt
import corefunction as corefunc
import folium
# Set page title


no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(no_sidebar_style, unsafe_allow_html=True)



# st.set_page_config(page_title="dashboard", menu_items={} , layout='wide')

# loading user data , profile, post and comment 
# profile_data = 


# st.markdown("""
#             <style> 
            
#                 .css-1v0mbdj{
#                     border: 1px solid red;
#                     padding:10px;                   
#                 }

#                 .css-1kyxreq{
#                     padding-left:25%;
#                 }


#             </style>
#     """, unsafe_allow_html=True)

# =================================== DATA LOADER =========================
# reading data 
def get_facebook_profiles():
    with open('profile.json', 'r') as file: 
        data = json.load(file)
    return data

# load model ..
# load model ....
model = joblib.load('naive_bayes_sentiment_analyzer_model.jb')

post_data = pd.read_csv('user_post_data.csv')
comment_table = pd.read_csv('comment_table.csv')
# st.write(post_data)
comment_table = comment_table[['post_id', 'comment_text', 'replies']]

facebookLoader = fl.FacebookLoader(get_facebook_profiles(),comment_table, post_data)
# st.write(comment_table)

# vulnerability score
_scores_text  = facebookLoader.personal_data_vulnerability_score(get_facebook_profiles(), facebookLoader.post_data['text'].values)
_scores_image_text = facebookLoader.personal_data_vulnerability_score(get_facebook_profiles(), facebookLoader.post_data['images_description'])  


final_post_table = facebookLoader.post_table_probability(_scores_text, _scores_image_text)

# ========================================================================================


# =================================== DATA LOADER =========================

comment_text_prob = facebookLoader.personal_data_vulnerability_score(get_facebook_profiles(), facebookLoader.comment_table['comment_text'].values)
comment_reply_prob = facebookLoader.personal_data_vulnerability_score(get_facebook_profiles(),  facebookLoader.comment_table['replies'].values)

# probability score
facebookLoader.comment_table_probability(comment_text_prob, comment_reply_prob)

# sentiment
_sentiment_com_text  = facebookLoader.sentiment_score(facebookLoader.comment_table['comment_text'].values, model)
_sentiment_reply_text  = facebookLoader.sentiment_score( facebookLoader.comment_table['replies'].values, model)

# adding sentiment column 
facebookLoader.comment_table['comment_sent'] = _sentiment_com_text 
facebookLoader.comment_table['reply_sent'] = _sentiment_reply_text 

post_test_corpus = []
# =================================== =========== =========================

def get_facebook_profile():
    with open('profile.json', 'r') as file: 
        data = json.load(file)
    
    return data


# fetching profile data
profile_info = get_facebook_profile()

# getting facebook post
def get_facebook_post(post_data):
    post = pd.read_csv(post_data,)
    return post


passport = Image.open('passport_image.jpg')
# print(passport)



def display_profile_data(profile_info): 
    p_id = profile_info['id']
    p_name = profile_info['name']
    p_mobile = profile_info['mobile']
    p_email = profile_info['email address']

    # st.button(str(p_id))
    st.write('Name      : {}'.format(str(p_name)))
    st.write('Mobile    : {}'.format(str(p_mobile)))
    st.write('Email     : {}'.format(str(p_email)))

def show_bar_chat(diction, title, xlabel, ylabel):
    adiction = {}
    for k,v in diction.items():
        adiction[k] = len(v) 

    fig , ax = plt.subplots()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.style.use('seaborn-v0_8-dark')
    ax.bar(adiction.keys(), adiction.values())
    st.pyplot(fig)
    plt.show()


def show_barh_chat(diction, title, xlabel, ylabel):
    adiction = {}
    for k,v in diction.items():
        adiction[k] = len(v) 

    fig , ax = plt.subplots()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.style.use('seaborn-v0_8-dark')
    ax.bar(adiction.keys(), adiction.values())
    st.pyplot(fig)
    plt.show()

def display_contac_match_content(df):
    st.write('==================================================')
    for d in range(len(df)):
        st.write('in {} ==|| ({} email) & ({} contact detected)'.format(df.iloc[d]['post_id'],df.iloc[d]['email_len'], df.iloc[d]['contact_len'] ))


def recommendation_prompt(score, tag):
    
    rec_text  = ''
    if (score <= 10) and (score >=0) :
        rec_text += f'{tag} is save from data exposure'  

    elif (score <= 30) and (score >= 10):
        rec_text += f'{tag} as minimum exposure of personal data'

    elif (score >= 30)  and (score <= 70): 
        rec_text += f'warning, {tag} content contain medium personal information'
        rec_text += f'\nRecommendation \n1. Delete {tag} \n2. Remove Sensitive information \n3. Secure Personal profile'

    else: 
        rec_text += f'{tag} content  reveal high personal information '
        rec_text += f'\nRecommendation \n1. Delete {tag} \n2. Remove Sensitive information \n3. Secure Personal profile'

    
    return rec_text

    



# ================================== end function ============================
with st.sidebar:
    st.subheader('FacebookAccount')
    st.image(passport, width=150)
    logout_button = st.button('logout')
    
    if logout_button: 
        switch_page_button.switch_page('main')

    
title_lay , button_lay = st.columns(2)
title_lay.subheader('Social Media Scrapper')




dash_layout, post_layout, comment_layout, googl_map_bar = st.tabs(['Dashboard', 'Post-Analysis', 'Comment-Analysis', 'Google-Map'])

with dash_layout:
    st.image(passport, width=200)
    st.metric("Facebook ID", str(profile_info['id']), '', delta_color='normal')
    # col1, col2 = st.columns([2,2])
    # col2.metric("Friends", str(profile_info['friend_count']), "")
    
    with st.expander('BIO DATA'):
        display_profile_data(profile_info)

    # with st.expander('RECOMMENDATION'):
    #     # display_profile_data(profile_info)
    #     POST = get_facebook_post('user_post_data.csv')





# user post analysis section......
with post_layout: 
    st.write('Post Data Analysis')
    # st.pyplot(plt.bar([1,2,3,4,5], [2,3,4,5,6]))
    post = get_facebook_post('user_post_data.csv')
    
    text_post = post.text.values
    image_post= post.images_description.values
    total_post = image_post +  text_post
    total_post = [tp for tp in total_post if tp != 'no content']
    post_test_corpus += total_post
    
    
    st.metric('Post Count...' , len(post) , delta_color='inverse')


    with st.expander('VISUAL ANALYSIS'):
        st.subheader('ENR VISUAL')
        
        totalpost_diction = corefunc.get_entities(total_post)
    
        show_bar_chat(totalpost_diction, 'Entity Analysis on Post & image', 'Entities', 'No of Entity')
        # st.write(totalpost_diction)

        

    with st.expander('VULNERABILITY ANALSYS'):
        st.subheader('User profile Exposure')
        fig , ax = plt.subplots()

        post_text_score = post['text_score(%)'].values
        post_image_score = post['images_description_score(%)'].values
        plt.xlabel('data')
        plt.ylabel('Vulnerability Probability (%)')
        plt.title('Analysis of User Post/image')
        ax.plot(post_text_score, label='post analysis')
        ax.plot(post_image_score, label='post image analysis')
        plt.legend()
        st.pyplot(fig)


    with st.expander('Email, Contact & Address Detector'):
        post_text = post.text.values[::-1]
        image_post = post.images_description.values[::-1]

         
            
        contact_data_detection = corefunc.contact_email_detector(post_text)
        image_data_detection = corefunc.contact_email_detector(image_post)
        # st.write(contact_data_detection)
        # st.write(image_data_detection)
        
        # lay_contact , lay_email = st.columns(2)
        contact_option_chk = st.checkbox('Post Text')
        if contact_option_chk: 
           display_contac_match_content(pd.DataFrame(contact_data_detection))

        email_option_chk  = st.checkbox('Post Image')
        if email_option_chk: 
            display_contac_match_content(pd.DataFrame(image_data_detection))


    with st.expander('POST 89SUMMARY TABLE'):
        
        st.write(facebookLoader.post_data.drop(columns=['post_id', 'text_score(%)', 'images_description_score(%)', 'final_score(%)']))

        post_d = facebookLoader.post_data
        recommendation_holder = []
        for index in post_d.index:
            # print(pt.iloc[index]['text'])
            score = post_d.iloc[index]['final_score(%)']
            recommendation_holder.append(recommendation_prompt(score, 'post'))
       
        post_d['Recommend'] = recommendation_holder
        st.write('system recommendation table')
        st.write(post_d[['Recommend', 'final_score(%)']])


        # val = facebookLoader.post_data['final_score(%)'].values
        # text = facebookLoader.post_data['text'].values 
        # image_text = facebookLoader.post_data['images_description'].values
        # texts = text + image_text
        # POST = facebookLoader.post_data

        # for post in range(len(POST)): 
        #     POST.iloc[post]

        # recommendation_prompt(POST)


# ============================  COMMENT LAYOUT =====================
with comment_layout: 
    st.write('Comment Data Analysis')
    com_table = corefunc.get_conmment_table()
    com_text = com_table.comment_text.values
    com_reply = com_table.replies.values
    total_comment = com_text +  com_reply
    total_comment = [tp for tp in total_comment if tp != 'no content']
    post_test_corpus += total_comment
    
    st.metric('Comment Count.' , len(com_table) , delta_color='inverse')


    with st.expander('VISUAL ANALYSIS'):
        st.subheader('ENR VISUAL')
        
        total_com_diction = corefunc.get_entities(total_comment)
    
        show_bar_chat(total_com_diction, 'Entity Analysis on Comment & Replies', 'Entities', 'No of Entity')
        # st.write(totalpost_diction)

        

    with st.expander('VULNERABILITY ANALSYS'):
        st.subheader('User profile Exposure')
        fig , ax = plt.subplots()

        comment_text_score = com_table['comment_text(%)'].values
        comment_reply_score = com_table['replies(%)'].values
        comment_total_score = com_table['final_score(%)'].values

        plt.xlabel('data')
        plt.ylabel('Vulnerability Probability (%)')
        plt.title('Analysis of User Post/image')
        ax.plot(comment_text_score, label='comment analysis')
        ax.plot(comment_reply_score, label='comment reply analysis')
        ax.plot(comment_total_score, label='comment tatal analysis')

        plt.legend()
        st.pyplot(fig)

    with st.expander('COMMENT SENTIMENT'):
        import numpy as np 
        sent_holder=[]
        comsent = [cs for cs in list(com_table['comment_sent'].values)]
        repsent = [rs for rs in list(com_table['reply_sent'].values)]
        
        sent_holder = np.array(comsent + repsent)
        unique_val , counts =np.unique(sent_holder, return_counts=True)
        figs , axs = plt.subplots()
        plt.xlabel('data')
        plt.ylabel('Sentiment Analysis')
        plt.title('Analysis on Comment & Reply')
        axs.barh(unique_val, counts)
        # plt.legend()
        st.pyplot(figs)

        

    with st.expander('Email, Contact & Address Detector'):
        comment_text = com_table.comment_text.values[::-1]
        reply_text = com_table.replies.values[::-1]

         
            
        contact_com_detection = corefunc.contact_email_detector(comment_text)
        email_com_detection = corefunc.contact_email_detector(reply_text)
        # st.write(contact_data_detection)
        # st.write(image_data_detection)
        
        # lay_contact , lay_email = st.columns(2)
        contact_option_chk = st.checkbox('Comment')
        if contact_option_chk: 
           display_contac_match_content(pd.DataFrame(contact_com_detection))

        email_option_chk  = st.checkbox('Replies')
        if email_option_chk: 
            display_contac_match_content(pd.DataFrame(email_com_detection))

    with st.expander('COMMENT TABLE'):
        
        st.write(com_table[['comment_text', 'replies', 'final_score(%)']])

        cm_table = com_table
        recommendation_holder_p = []
        for index in com_table.index: 
            score = com_table.iloc[index]['final_score(%)']
            recommendation_holder_p.append(recommendation_prompt(score, 'comment'))
        
        
        cm_table['Recommend'] = recommendation_holder_p

        st.write('Recommendation Table')
        st.write(cm_table[['Recommend' , 'final_score(%)']])

        

with googl_map_bar: 

    option_list = []
    entity_find = corefunc.get_entities(post_test_corpus)
    # st.write(entity_find)

    if entity_find.get('GPE') != None:
        option_list += entity_find.get('GPE')
    
    if entity_find.get('NORP') != None: 
        option_list += entity_find.get('NORP')
    
    option_list = set(option_list)
    options = st.selectbox(f'{len(option_list)} Location Found', option_list)


    if options: 
        google_data, loc_size = None , 0  
        with st.spinner('Searching'):
            try:
                google_data, loc_size = corefunc.get_google_location(options)

            except:
                st.warning('Error Connecting')

            st.subheader('AI Detect {} Location for {}'.format(loc_size, options))

            # st.write('Location {} : address {}'.format(1, google_data['name'][0]))
            lon , lat = google_data['lon'] , google_data['lat']

            df_locations  = pd.DataFrame(google_data)
            df_locations.lat = df_locations.lat.astype('float32')
            df_locations.lon = df_locations.lon.astype('float32')

            # for index, row in df_locations.iterrows():
            #     folium.Marker([row['lat'], row['lon']], popup=row['name']).add_to(st.pydeck_chart())
            
            st.write(df_locations)
        
            st.map(df_locations[['lat', 'lon']])
            
    # st.write(post_test_corpus)
            
 
