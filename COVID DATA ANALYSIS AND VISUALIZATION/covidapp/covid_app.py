import streamlit as st 
import pandas as pd
import re
import matplotlib.pyplot as plt
import datetime 
from epiweeks import Week as wk
from datetime import date
import datetime as dt 
from datetime import date, timedelta
import numpy as np
import re


st.header('Covid Case Analysis')


# importing dataset 
covid = pd.read_csv('updated_covid_record.csv')

covid['date'] = pd.to_datetime(covid["date"], format="%Y/%m/%d")

def extract_month_name(date):
#     datetime.datetime.strptime(date.month, '%m')
#     month = date.strftime("%b")
    month = date.strftime("%b")
    return month

def extract_week_of_the_year(date):
    wek = wk.fromdate(date)
    wek.week
    wek_str = f'Week {wek.week}'
    return wek_str


def extract_week_count(date):
    wek = wk.fromdate(date)
#     wek.week
#     wek_str = f'Week {wek.week}'
    return wek.week

def return_string_date_format(date):
    full_date = date.strftime("%d %b %Y")
    return full_date


covid['full_date'] = covid['date'].apply(lambda d:return_string_date_format(d))


#  geting date range coumns.....
d1, d2 = st.columns([1,1])

start_date = d1.date_input(value = covid.date[0], label='Start Date' ,min_value=covid.date.min(), max_value=covid.date.max())
end_date = d2.date_input(value = covid.date[15], label='End Date' ,min_value=covid.date.min(), max_value=covid.date.max())


def get_date_and_index_dictionary(start_date, end_date, covid):
#     init_date = '2020-03-18' 
    init_date = start_date.strftime('%Y-%m-%d')
    # getting the date range... 
    # d1_index = covid.index[covid['date'] == "2020-03-18" ]
    # d2_index = covid.index[covid['date'] == "2020-03-25" ]
    d1_index = covid.index[covid['date'] == start_date.strftime('%Y-%m-%d') ]
    d2_index = covid.index[covid['date'] == end_date.strftime('%Y-%m-%d')   ]

    day_different = covid.date[d2_index[0]]- covid.date[d1_index[0]]
    days = day_different.days
    start_index = d1_index[0]
    end_index = start_index + days

    date_dictionary = {}
    date_index = []
    for d in range(days+1):
        date_dictionary[init_date] = d1_index
        date_index.append(d1_index)
        dt_obj = dt.datetime.strptime(init_date, '%Y-%m-%d')
        dt_obj += timedelta(days=1)
        new_index = covid.index[covid['date'] == dt_obj.strftime('%Y-%m-%d') ]
        d1_index = new_index
        init_date = dt_obj.strftime('%Y-%m-%d')


    # for key , indexes in date_dictionary.items():
    #     print(f'date {key} val {len(indexes)}')

    st.subheader(f'Covid Case over : {days} days')
    # st.write(f'start day : {start_index}')
    # st.write(f'end day :{end_index}')
    
    return date_dictionary ; 



def get_total_case_for_a_date(col, date_index): 
    
    #     print(index)
    two_d = covid[col].values
    condi = range(len(two_d))
    sum_val = [d.sum() for d in two_d]
    d1_list = list(date_index)
    mch_val = []
    for d in d1_list:
        mch_val.append(sum_val[d])

    total_case = np.array(mch_val).sum()
    return total_case

case_indexSize = get_date_and_index_dictionary(start_date, end_date, covid)

# creating dictionary of variou covid case for a particular date... 
date_dictionary2 = {}
for key , indexes in case_indexSize.items():

    covid_cal_col = [col for col in covid.columns if  re.search('newCasesBySpecimenDateRollingSum|newCasesBySpecimenDate-', col)]
    date_dictionary2[key] = get_total_case_for_a_date(covid_cal_col, indexes)

# st.write(date_dictionary2)
def show_linechart():
    X = []
    Y = []
    for key , indexes in date_dictionary2.items():
        # print(f'date {key} val {indexes}')
        X.append(key)
        Y.append(indexes)
    
    st.line_chart(Y, x = X)

show_linechart()
# plt.plot(range(len(cd_cases)) , cd_cases)
# plt.show()


# ======================================== BAR CHART FOR WEEKLY VISUALIZATION ============================================
extracted_date = [dt.datetime.strptime(key, '%Y-%m-%d') for key ,index in case_indexSize.items()]
week_count = [extract_week_count(ex) for ex in extracted_date]
week_count = np.unique(np.array(week_count))
week_count


def get_weekly_case_info(weeks, col):
    week_diction = {}
    for week in weeks:
        weekly_total_case = covid[col][covid['week_count'] == week].sum().values.sum()
        # print(weekly_total_case)
        week_diction[f'week {week}']  = weekly_total_case
    
    return week_diction


weekly_data_info = get_weekly_case_info(week_count, covid_cal_col)


# fig, ax = plt.subplots()
# ax.bar(weekly_data_info.keys(), weekly_data_info.values())
# st.pyplot(fig)

week_df = pd.DataFrame(data= [weekly_data_info.values()] , columns=weekly_data_info.keys())

st.bar_chart(weekly_data_info.values() )






