import streamlit as st
from streamlit_chat import message
import pandas as pd
import string as stng
import random
import re
import sqlite3 as sqlite
# loading dataset

st.markdown(''' 
        <h1 align='center'>CYBER BOT </h1>
    ''', unsafe_allow_html=True)


class Database(): 

    def __init__(self) -> None:
        self.dbconnect = sqlite.connect('database.db')

    def create_history_table(self): 
        query = """
        CREATE TABLE IF NOT EXISTS query_table(
            query CHAR(200) NOT NULL, 
            respond TEXT NOT NULL
        )
        """
        self.dbconnect.execute(query)
        print('table created successfully... ')


    def insert_history(self, user_input, respond): 

        query = 'INSERT INTO query_table VALUES(?,?)'
        size = len(self.show_history())
        self.dbconnect.execute(query,(user_input, respond) )
        self.dbconnect.commit()

    def show_history(self): 
        query = 'SELECT * FROM query_table'
        query_action = self.dbconnect.execute(query)
        
        return  query_action.fetchall()


# init databased
database = Database()
database.create_history_table()

print(database.show_history())





item = None
with open('doc.txt', 'r', encoding='utf8') as file: 
    item = (file.readlines())
    

def clean_and_process_data(data): 
    data = [d.replace('\n', '') for d in data]

    return data

clean_doc = clean_and_process_data(item)


question = clean_doc[::2]
answers = clean_doc[1::2]
import pandas as pd 

dataset = {
    'question': question, 
    'answers': answers
}

bot_dataset = pd.DataFrame(dataset)


text = bot_dataset.question[100]
bot_dataset['clean_question'] =  bot_dataset.question.apply(lambda x: re.split("-|\s", x.lower().strip())) 
bot_dataset = bot_dataset[:-1]

# ________________________________________________________________________
# ________________________________________________________________________




message('hello am cyber bot, how can i help you?')


class LongResponse: 

    def __init__(self) -> None:
        pass
    
    def unknown(self): 
        return ['canot understand you word' , 'i dont understand', 'what do you mean', '.....'][random.randrange(4)]

class LongResponse: 

    def __init__(self) -> None:
        pass
    
    def unknown(self): 
        return ['canot understand you word' , 'i dont understand', 'what do you mean', '.....'][random.randrange(4)]


LR = LongResponse()

# method for removing noise from data.. 
def clean_input(data):
    # removing punctuation
    data = ''.join([i for i in data if not i in  stng.punctuation])
    #splitting data into list
    data = data.split()
    # converting text into lower case
    data = [i.lower() for i in data]

    return data


# method to compute responce probability 
def message_probability(user_query, recognize_words, single_response=False, candidate_words=[]):
    message_certainity = 0
    has_required_words = True
    
    # print('user query' , user_query)
#     going throught user words 
    for term in user_query: 
        if term in recognize_words: 
            message_certainity = message_certainity + 1 
    
#     calculating percentage probability 
    percentatege = float(message_certainity) / float(len(recognize_words))  
    # print('percentage : ', percentatege)
    
#     checking through the require word 
    for word in candidate_words:
        # print('searching words...... ' , word)
        
        if word  in user_query: 
            has_required_words = True
            print('REQUIRE FOUND : ', word )
            break
    else: 
        has_required_words = False
        # print('always')
    
    # print("has required : {} , singel respose : {}".format(has_required_words, single_response))
    if has_required_words or single_response: 
        return int(percentatege*100)
    else: 
        return 0


def check_all_messages(message): 
    
    # we have (responce of bot ======= MAP ====== probability_score) ---- DICTIONARY...
    highest_prob_list = {}
    
    def response(bot_response, list_of_words, single_response=False, require_word=[]):
        nonlocal highest_prob_list
        highest_prob_list[bot_response] = message_probability(message, list_of_words, single_response, require_word)
    
#   Response..................................................................
    response('hello', ['hello', 'hi', 'sup', 'hey', 'heyo'], single_response=True )
    response('I\'m doing fine, and you?', ['how', 'are', 'you', 'doing'], require_word=['how'])
    response('Thank you', ['i', 'love', 'code', 'palace'], require_word=['code', 'palace'] )
    
    for index in range(len(bot_dataset)):
        
        qt = bot_dataset.iloc[index].clean_question 
        ans = bot_dataset.iloc[index].answers
        print('{} Message: {}'.format(qt, message))
        response(ans, qt, require_word=qt)
    
                                                                                               
    best_match = max(highest_prob_list, key=highest_prob_list.get)
    print(highest_prob_list.values())
    

    return  LR.unknown() if  highest_prob_list[best_match] < 1  else best_match
    
                                                                                           
                                                                                        

# method to get bot responces
def get_response(user_input): 
    
    # cleaning of user input
    clean_query = clean_input(user_input)
    rel = check_all_messages(clean_query)
    # print(rel)

#     cleaning of user input
    return rel


# ___________________________________
placeholder = st.empty()
# query = input('You:  ')
query = st.text_input('You: ')
query = query.lower()
# st.write(query)
message_history = {
    "message":[], 
    "is_bot":[]
}

if query: 
    # if query == 'exit':
    #     message('good bye')

    response = get_response(str(query))
    # print('responce: ' , response)
   
    # saving message history for user 
    message_history['message'].append(query)
    message_history['is_bot'].append(True)

    # saving message history for bot 
    message_history['message'].append(response)
    message_history['is_bot'].append(False)
    # updating screen
    with placeholder.container():

       
        database.insert_history(query, response)

        # message(f'{response}')
        print(len(message_history['message']))

        for index in range(len(message_history['message'])): 

            msg = message_history['message'][index]
            true_false = message_history['is_bot'][index]

            # print('{} {}'.format(msg, true_false))
            message(msg, is_user=true_false , key=f'{str(random.randint(1, 50000))}')
            st.empty()


    


