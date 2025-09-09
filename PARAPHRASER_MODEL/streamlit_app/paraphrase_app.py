import streamlit as st
from simpletransformers.t5 import T5Model
import os
import database as db 
import sqlite3 as sq


st.set_page_config(layout='wide')


# ++++++++++++++++++++++++++++++++++++++++++++++++=  start database functionality ===============================

with sq.connect('paraphrasedb.db') as db: 
    database = db


def create_history_table():
    database.execute('''
                CREATE TABLE IF NOT EXISTS paraphrase_table  (
                    id INTEGER AUTO INCREMENT, 
                    original_text TEXT NOT NULL, 
                    praraphrase_text TEXT NOT NULL
                ) 
    ''')

    # st.success('table created successfully.. ')


def insert_into_history(ot , pt):
    query ='''insert into paraphrase_table values (? ,? , ?)
    '''
    result = database.execute(query, ( 1, ot, pt))
    database.commit()
    # st.success("history recorded successfully")

def get_database_history(): 
    query =''' select * from paraphrase_table

    '''
    rows = database.execute(query).fetchall()
    return rows 



create_history_table()
# ================================= end dataabse functionality ====================================


# ================================================
# trained_model_path = os.path.join(root_dir,"outputs")

def paraprashe_text(text, return_sequence=3):
    args = {
    "overwrite_output_dir": True,
    "max_seq_length": 256,
    "max_length": 50,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": return_sequence
    }
    trained_model_path =  'checkpoint-2064-epoch-4'
    trained_model = T5Model("t5",trained_model_path,args=args, use_cuda=False)

    prefix = "paraphrase"
    pred = trained_model.predict([f"{prefix}: {text}"])

    print('Paraphrased')
    return pred[0]
# ================================================

st.markdown('<i><h2>T-5 Paraphraser Tools </h2><i>' , unsafe_allow_html=True)

# with st.spinner("Model loading"):
rephrase_tab , history_tab = st.tabs(['rephraser' , 'history'])

with rephrase_tab:
    # defining layout .... row one....
    original_col , paraphrased_col = st.columns([3,3])

    # originl text and pase area widget... 
    print(st.session_state)
    pc = paraphrased_col.empty()
    txt_area = original_col.text_area('' , placeholder='Original Tezt' , height=300,label_visibility='collapsed')

    

    txt_gen_area = pc.text_area('', placeholder='Paraphase Text' , height=300, label_visibility='collapsed')

    # row2 layout 
    # number of return
    lay , lay1, lay2 = st.columns([2 ,3, 1])


    if lay2.button('Paraphrase'):
        with st.spinner("Paraphrasing"):
            text = paraprashe_text(txt_area, 2)
            final_text = text[-1]
            txt_gen_area = pc.text_area('', f'{final_text}', placeholder='Paraphase Text' , height=300, label_visibility='collapsed')
            insert_into_history(txt_area, final_text)


    

with history_tab: 
    records = get_database_history()
    
    t1 , t2   = st.columns([3,3])
    t1.subheader("Original Text")
    t2.subheader("Paraphrase Text")
    for re in records:
        or_text , re_text = re[1] , re[2] 
        origin , rephrase = st.columns([3,3])
        st.markdown('<hr>', unsafe_allow_html=True)

        origin.write(or_text) 
        rephrase.write(re_text)
      
        
