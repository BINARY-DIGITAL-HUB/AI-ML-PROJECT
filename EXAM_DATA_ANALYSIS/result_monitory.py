import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sqlite3 as sq

st.header('Acedamic Performance Monitoring System')


imt = pd.read_csv("imt_data.csv")
css = pd.read_csv("css_data.csv")
com = pd.read_csv("com_data.csv")



class Database: 
    # database: None 
    def __init__(self) :
        # self.database
        with sq.connect('exam_database.db') as db: 
            self.database = db



    def create_student_table(self):

        self.database.execute('''
                    CREATE TABLE IF NOT EXISTS student_table (
                        email primary key, 
                        matric_number TEXT NOT NULL
                    ) 
        ''')

    def create_biodata_table(self):

        self.database.execute('''
                    CREATE TABLE IF NOT EXISTS bio_table (
                        dept TEXT NOT NULL,
                        sex TEXT NOT NULL, 
                        age INT NOT NULL,
                        name VARCHAR NOT NULL, 
                        entry_year INT NOT NULL, 
                        matric  TEXT NOT NULL
                    ) 
        ''')

    def create_exam_record_table(self):

        self.database.execute('''
                    CREATE TABLE IF NOT EXISTS bio_table (
                        level_100 TEXT NOT NULL,
                        level_200 VARCHAR NOT NULL, 
                        level_300 VARCHAR NOT NULL,
                        level_400 VARCHAR NOT NULL
                    ) 
        ''')


    def verify_student(self, email, matric):

        query = 'select email, matric_number from student_table' 
        records  = self.database.execute(query).fetchall()
       
        marker = False 
        for re in records: 
            # st.write(re)
            if re[0].lower() == matric.lower() and re[1].lower() == email.lower() : 
                # st.write(f'{email} {matric} {re[0]}{re[1]}')
                marker = True

        return marker 
 
# ==================== initiating database =======

database  = Database()
database.create_student_table()
database.create_biodata_table()
database.create_exam_record_table()


# ======================== end ===================



def get_level_data(dept=com , level = 100 , start=3, stop=7):
    
    data = None 
    
    if level == 100:    
        data = dept[dept.columns[4:8]]
    elif level == 200: 
         data = dept[dept.columns[8:12]]
    elif level == 300: 
        data = dept[dept.columns[12:16]]
    elif level == 400:
       data = dept[dept.columns[16:20]]
    else : 
        data = None
        
    return data

# level_data = get_level_data(com , 100)[:10]
# st.dataframe(level_data)
def cover_score_to_grade(score):
    
    if score >= 0 and score < 40 : 
        grade = 'F'
    elif score >= 40 and score < 49 : 
        grade = 'E'
    elif score >= 50 and score < 60 :
        grade = 'C'
    elif score >= 60 and score < 75 :
        grade = 'B'
    elif score >= 75 and score <= 100 : 
        grade = 'A'
    else: 
        grade='F'
        
    return grade


def convert_to_grade_table(level_data):

    level_grade_list =  []
    for index in range(len(level_data)): 
        d=level_data.values[index]
        grade = [cover_score_to_grade(dm) for dm in d] 
        level_grade_list.append(grade)
    #     print(grade)

    level_grade_list = np.array(level_grade_list)
    df = pd.DataFrame(level_grade_list , columns=level_data.columns )
    return df


# data = convert_to_grade_table(level_data)

def get_course_score_statistic(data, course): 
    
    grade_list = ['A', 'B', 'C' , 'D', 'E', 'F']
    diction_data = {}
    diction_percentatage = {}
    total_count = []  

    for index in range(len(grade_list)):
                       
        count = data[course][data[course]== grade_list[index]].count()
        diction_data[grade_list[index]] = count
        total_count.append(count)

    
    total_count= np.array(total_count)
#     print(total_count.sum())
    percent_distribution = [(cnt/total_count.sum()) * 100 for cnt in total_count]
    
    
    return diction_data , percent_distribution
                       
                       
# get_course_score_statistic(data, 'COM314')  
# 


with st.sidebar: 

    st.header('Menu')

    option = st.selectbox('DEPARTMENT' , options=['COM SCI', 'IMT', 'CSS'])


    level_option = st.radio('LEVEL' , ['100', '200', '300', '400'] ,  horizontal=True )
    level_data = get_level_data(com , 100)[:10]
    # st.selectbox('COURSE' , level_data.columns)

    st.write('______________________________')
    st.write('Check Exams Data.. ')
    em = st.text_input("School Email")
    mt_no = st.text_input("Matric No.")
    
    personal_record_activat = False
    if st.button('check record'): 
        marker = database.verify_student(em, mt_no)
        if marker: 
            st.success('Successful')
            personal_record_activat = True

            
        else: 
            st.error('student not found')


# =================================  INNNER METHOD 

def check_record(matric):

    record:None 
    if matric in imt['matric'].values: 
        print('imt')
        record  = imt[imt['matric'] ==  matric]
    elif matric in css['matric'].values: 
        print('css')
        record  = css[css['matric'] ==  matric]
    elif matric in com['matric'].values: 
        print('cpt')
        record  = com[com['matric'] ==  matric]
    else: 
        record = None
    
    st.table(record[record.columns[4:]])
    st.table(convert_to_grade_table(record[record.columns[4:]]))
    
    return record.values


def get_record_detail(record):
    data = {}
    
    data['dept'] = record[0][1]
    data['matric'] = record[0][2]
    data['name'] = record[0][3]
    data['level1'] = record[0][4:8]
    data['level2'] = record[0][8:12]
    data['level3'] = record[0][12:16]
    data['level4'] = record[0][16:20]
    
    return data



# ==============================================   HELPER FUNCTION  =====================


if  personal_record_activat : 
   
    detail , record_view = st.columns([3,3])

    
    
    

    # look up data .. 
    record = check_record(str(mt_no).lower())
    # st.write(record)
    the_record = get_record_detail(record)
    # st.write(the_record)

    detail.write(f"Department :  {the_record['dept']}")
    detail.write(f"Name :  {the_record['name']}")
    detail.write(f"Matric :  {the_record['matric']}")

    
    if st.button('Back'): 
        personal_record_activat == False


else: 

    if option == 'SICT': 
        pass    
    elif option == 'COM SCI': 
        
        st.subheader(f'{option}  {level_option} LEVEL')

        col1, col2, col3 , col4, col5, col6= st.columns(6)
        # col1.metric(option, "A", "1.2 °F")
        # col2.metric(option, "B", "8%")
        # col3.metric(option, "C", "4%")
        # col4.metric(option, "D", "4%")
        # col5.metric(option, "E", "-4%")
        # col6.metric(option, "F", "-4%")
        
        level_data = get_level_data(com , int(level_option))[:20]
        selected_course = st.selectbox('COURSE' , level_data.columns)
        v1, v2 = st.columns([3,3])



        with v1.expander('SHOW GRADE'):

            
            grade_df = convert_to_grade_table(level_data)
            grade_diction ,percent_distribution = get_course_score_statistic(grade_df, str(selected_course))
            
            # ==========================
            val = list(grade_diction.values())
            # st.write(val[0])
            col1.metric(option, f"A ({int(val[0])})", f'{int(percent_distribution[0])}%')
            col2.metric(option, f"B ({int(val[1])})", f'{int(percent_distribution[1])}%')
            col3.metric(option, f"C ({int(val[2])})",f'{int(percent_distribution[2])}%')
            col4.metric(option, f"D ({int(val[3])})", f'{int(percent_distribution[3])}%')
            col5.metric(option, f"E ({int(val[4])})", f'{int(percent_distribution[4])}%' , delta_color='inverse')
            col6.metric(option, f"F ({int(val[5])})", f'{int(percent_distribution[5])}%' , delta_color='inverse')

            # ==========================

            # x_axis = np.range(len(grade_diction.keys()))

            fig, ax = plt.subplots()
            # plt.figure(figsize=(2,4))
            ax.bar(grade_diction.keys(), grade_diction.values(), color=['green', 'brown', 'yellow', 'blue', 'gold', 'red'])
            # ax.bar(grade_diction.keys(), grade_diction.values())
            
            st.pyplot(fig)

        with v2.expander('VIEW PERCENTAGE DISTRIBUTION'):

            # selected_course = st.selectbox('SELECT COURSE.' , level_data.columns)
            grade_df = convert_to_grade_table(level_data)
            grade_diction ,percent_distribution = get_course_score_statistic(grade_df, str(selected_course))
            

            # x_axis = np.range(len(grade_diction.keys()))

            fig, ax = plt.subplots()
            ax.pie(grade_diction.values(), labels=grade_diction.keys(), autopct='%1.1f%%')
            # ax.bar(grade_diction.keys(), grade_diction.values())
            
            st.pyplot(fig)
        
        # with st.expander('VIEW TABLE DISTRIBUTION'): 
        #     st.write('HELLO')
        

    elif option == 'IMT':
        
        st.subheader(f'{option}  {level_option} LEVEL')

        col1, col2, col3 , col4, col5, col6= st.columns(6)
        # col1.metric(option, "A", "1.2 °F")
        # col2.metric(option, "B", "8%")
        # col3.metric(option, "C", "4%")
        # col4.metric(option, "D", "4%")
        # col5.metric(option, "E", "-4%")
        # col6.metric(option, "F", "-4%")
        
        level_data = get_level_data(com , int(level_option))[:20]
        selected_course = st.selectbox('COURSE' , level_data.columns)
        v1, v2 = st.columns([3,3])

        with v1.expander('SHOW GRADE'):

            
            grade_df = convert_to_grade_table(level_data)
            grade_diction ,percent_distribution = get_course_score_statistic(grade_df, str(selected_course))
            
            # ==========================
            val = list(grade_diction.values())
            # st.write(val[0])
            col1.metric(option, f"A ({int(val[0])})", f'{int(percent_distribution[0])}%')
            col2.metric(option, f"B ({int(val[1])})", f'{int(percent_distribution[1])}%')
            col3.metric(option, f"C ({int(val[2])})",f'{int(percent_distribution[2])}%')
            col4.metric(option, f"D ({int(val[3])})", f'{int(percent_distribution[3])}%')
            col5.metric(option, f"E ({int(val[4])})", f'{int(percent_distribution[4])}%' , delta_color='inverse')
            col6.metric(option, f"F ({int(val[5])})", f'{int(percent_distribution[5])}%' , delta_color='inverse')

            # ==========================

            # x_axis = np.range(len(grade_diction.keys()))

            fig, ax = plt.subplots()
            # plt.figure(figsize=(2,4))
            ax.bar(grade_diction.keys(), grade_diction.values(), color=['green', 'brown', 'yellow', 'blue', 'gold', 'red'])
            # ax.bar(grade_diction.keys(), grade_diction.values())
            
            st.pyplot(fig)

        with v2.expander('VIEW PERCENTAGE DISTRIBUTION'):

            # selected_course = st.selectbox('SELECT COURSE.' , level_data.columns)
            grade_df = convert_to_grade_table(level_data)
            grade_diction ,percent_distribution = get_course_score_statistic(grade_df, str(selected_course))
            

            # x_axis = np.range(len(grade_diction.keys()))

            fig, ax = plt.subplots()
            ax.pie(grade_diction.values(), labels=grade_diction.keys(), autopct='%1.1f%%')
            # ax.bar(grade_diction.keys(), grade_diction.values())
            
            st.pyplot(fig)
        # st.dataframe(level_data)
        

    else:
        st.subheader(f'{option}  {level_option} LEVEL')

        col1, col2, col3 , col4, col5, col6= st.columns(6)
        # col1.metric(option, "A", "1.2 °F")
        # col2.metric(option, "B", "8%")
        # col3.metric(option, "C", "4%")
        # col4.metric(option, "D", "4%")
        # col5.metric(option, "E", "-4%")
        # col6.metric(option, "F", "-4%")
        
        level_data = get_level_data(com , int(level_option))[:20]
        selected_course = st.selectbox('COURSE' , level_data.columns)
        v1, v2 = st.columns([3,3])

        with v1.expander('SHOW GRADE'):

            
            grade_df = convert_to_grade_table(level_data)
            grade_diction ,percent_distribution = get_course_score_statistic(grade_df, str(selected_course))
            
            # ==========================
            val = list(grade_diction.values())
            # st.write(val[0])
            col1.metric(option, f"A ({int(val[0])})", f'{int(percent_distribution[0])}%')
            col2.metric(option, f"B ({int(val[1])})", f'{int(percent_distribution[1])}%')
            col3.metric(option, f"C ({int(val[2])})",f'{int(percent_distribution[2])}%')
            col4.metric(option, f"D ({int(val[3])})", f'{int(percent_distribution[3])}%')
            col5.metric(option, f"E ({int(val[4])})", f'{int(percent_distribution[4])}%' , delta_color='inverse')
            col6.metric(option, f"F ({int(val[5])})", f'{int(percent_distribution[5])}%' , delta_color='inverse')

            # ==========================

            # x_axis = np.range(len(grade_diction.keys()))

            fig, ax = plt.subplots()
            # plt.figure(figsize=(2,4))
            ax.bar(grade_diction.keys(), grade_diction.values(), color=['green', 'brown', 'yellow', 'blue', 'gold', 'red'])
            # ax.bar(grade_diction.keys(), grade_diction.values())
            
            st.pyplot(fig)

        with v2.expander('VIEW PERCENTAGE DISTRIBUTION'):

            # selected_course = st.selectbox('SELECT COURSE.' , level_data.columns)
            grade_df = convert_to_grade_table(level_data)
            grade_diction ,percent_distribution = get_course_score_statistic(grade_df, str(selected_course))
            

            # x_axis = np.range(len(grade_diction.keys()))

            fig, ax = plt.subplots()
            ax.pie(grade_diction.values(), labels=grade_diction.keys(), autopct='%1.1f%%')
            # ax.bar(grade_diction.keys(), grade_diction.values())
            
            st.pyplot(fig)
        # st.dataframe(level_data)
        


    # ==============================================   HELPER FUNCTION  =====================
