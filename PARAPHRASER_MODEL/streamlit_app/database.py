# with sq.connect('paraphrasedb.db') as db: 
#     database = db


# def create_history_table():
#     database.execute('''
#                 CREATE TABLE IF NOT EXISTS paraphrase_table  (
#                     id auto increatment primary key, 
#                     original_text TEXT NOT NULL, 
#                     praraphrase_text TEXT NOT NULL
#                 ) 
#     ''')

#     # st.success('table created successfully.. ')


# def insert_into_history(ot , pt):
#     query = 'insert into paraphrase_table values (?,?)'
#     database.execute(query, (ot, pt))
#     # st.success("history recorded successfully")
