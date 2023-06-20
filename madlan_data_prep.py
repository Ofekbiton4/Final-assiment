# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 18:17:05 2023

@author: Ofek biton & Shahaf Malka
"""

import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib as plt
from datetime import datetime


excel_file = 'output_all_students_Train_v10.xlsx'
data = pd.read_excel(excel_file)



def update_price(dataframe):
    dataframe['price'] = dataframe['price'].astype(str)
    dataframe['price'] = dataframe['price'].apply(lambda x: re.sub('[^\d\.]+','',x))
    dataframe['price'] = pd.to_numeric(dataframe['price'], errors='coerce').dropna().astype(int)
    dataframe = dataframe.dropna(subset=['price'] , axis=0, inplace=True)
    
def update_type(dataframe):
    dataframe['type'] = dataframe['type'].replace({
        'דירה': 'דירה בבניין',
        'בניין': 'דירה בבניין',
        'דירת גג': 'דירה בבניין',
        'דו משפחתי': 'דו משפחתי',
        'דופלקס': 'דו משפחתי',
        'פנטהאוז': 'פנטהאוז',
        'מיני פנטהאוז': 'פנטהאוז',
        'דירת נופש': 'בית פרטי',
        "קוטג'": 'בית פרטי',
        "קוטג' טורי": 'בית פרטי',
        'מגרש': 'אחר',
        'נחלה': 'אחר',
        'טריפלקס': 'אחר'
    })
    value_to_drop = ['אחר']
    dataframe = dataframe.drop(dataframe[dataframe['type'].isin(value_to_drop)].index)

def update_Area(dataframe):
    for index, row in dataframe.iterrows():
        area = row['Area']
        if isinstance(area, int) or isinstance(area, float):
            dataframe.at[index, 'Area'] = area
        else:
            try:
                area_match = re.findall(r'\d+', str(area))
                area_match = int(area_match[0])
                dataframe.at[index, 'Area'] = area_match
            except:
                dataframe.at[index, 'Area'] = None
    dataframe.dropna(subset=['Area'], inplace=True)
    dataframe['Area'] = dataframe['Area'].astype(float)
    
def update_room_number(dataframe):
    dataframe['room_number'] = dataframe['room_number'].astype(str)
    dataframe['room_number'] = dataframe['room_number'].apply(lambda x: re.sub(r'[^\d.]', '', x))
    dataframe['room_number'] = dataframe['room_number'].replace('', np.nan)
    dataframe['room_number'] = dataframe['room_number'].astype(float)
    dataframe.loc[dataframe['room_number'] > 10, 'room_number'] = np.nan
    dataframe.dropna(subset=['room_number'], inplace=True)
    
def clean_strings(dataframe):
    dataframe['Street'] = dataframe['Street'].str.replace('[^\w\s]', '', regex=True)
    dataframe['city_area'] = dataframe['city_area'].str.replace('[^\w\s]', '', regex=True)
    dataframe['description'] = dataframe['description'].str.replace(r'\W+', ' ', regex=True)
    for index, row in dataframe.iterrows():
        dataframe.loc[index, 'City'] = row['City'].strip()
    dataframe['City']=dataframe['City'].replace(' נהרייה','נהריה')
    dataframe['City']=dataframe['City'].replace('נהרייה','נהריה')
    dataframe['City']=dataframe['City'].replace(' שוהם','שוהם')
    dataframe['City']=dataframe['City'].replace('שוהם','שוהם')
    dataframe['condition'] = dataframe['condition'].replace(['None', False], 'לא צויין')
    dataframe['condition'] = dataframe['condition'].replace('דורש שיפוץ', 'ישן')
    
def unpdate_floor_data(dataframe):
    for index, row in dataframe.iterrows():
        floor_value = row['floor_out_of']
        if pd.isna(floor_value):
            continue
        floors = str(floor_value).lstrip().split(" ")
        try:
            if 'מרתף' in floors:
                dataframe.at[index, 'floor'] = -1
                dataframe.at[index, 'total_floors'] = 2
            elif 'קרקע' in floors:
                dataframe.at[index, 'floor'] = 0
                dataframe.at[index, 'total_floors'] = 0
            else:
                dataframe.at[index, 'floor'] = floors[1]
                dataframe.at[index, 'total_floors'] = floors[3]
        except IndexError:
            pass
    dataframe.dropna(subset=['total_floors','floor'], inplace=True)
    dataframe['floor'] = dataframe['floor'].astype(float)
    dataframe['total_floors'] = dataframe['total_floors'].astype(float)
    
def process_entrance_date(dataframe):
    for index, row in dataframe.iterrows():
        val = row['entranceDate']
        try:
            entrance_time = (val - datetime.now()).days
            if entrance_time < 183:
                dataframe.at[index, 'entrance_Date'] = 'less_than_6_months'
            elif entrance_time > 365:
                dataframe.at[index, 'entrance_Date'] = 'above_year'
            else:
                dataframe.at[index, 'entrance_Date'] = 'months_6_12'
        except:
            if pd.isna(val) or "לא צויין" in val or val == False:
                dataframe.at[index, 'entrance_Date'] = "not_defined"
            if "גמיש" in val or "flexible" in val:
                dataframe.at[index, 'entrance_Date'] = "flexible"
            if "מיידי" in val or "immediate" in val:
                dataframe.at[index, 'entrance_Date'] = "less_than_6_months"
                
def update_bool_column(dataframe):
    columns_to_convert = ['hasElevator','hasParking','hasBars','hasStorage','hasAirCondition','hasBalcony','hasMamad','handicapFriendly']
    mapping = {'כן': 1,'True': 1,'Yes': 1,'יש': 1,'לא': 0,'no': 0,'False': 0,'אין': 0, np.nan: 0}
    patterns_true = r'\b(?:כן|true|yes|יש|נגיש)\b'
    patterns_false = r'\b(?:לא|no|false|אין|nan)\b'
    for column_name in columns_to_convert:
        dataframe[column_name] = dataframe[column_name].apply(lambda x: 1 if re.search(patterns_true, str(x), flags=re.IGNORECASE) else 0 if re.search(patterns_false, str(x), flags=re.IGNORECASE) else mapping.get(str(x).lower(), x))
        dataframe[column_name] = dataframe[column_name].replace(mapping).astype(int)
        
def change_column_place(dataframe):
    column_name = 'price'
    column = dataframe['price']
    dataframe.pop(column_name)
    dataframe[column_name] = column
    dataframe.drop(['number_in_street','Street','city_area', 'floor_out_of','num_of_images','entranceDate','publishedDays','description','condition','floor','total_floors','entrance_Date','hasElevator','hasBars','furniture','hasBalcony','hasStorage'], axis = 1, inplace = True)
    

def prepare_data(dataframe):
    #dataframe = dataframe.copy()
    dataframe.columns = dataframe.columns.str.strip()
    update_price(dataframe)
    update_type(dataframe)
    update_Area(dataframe)
    clean_strings(dataframe)
    update_room_number(dataframe)
    unpdate_floor_data(dataframe)
    process_entrance_date(dataframe)
    update_bool_column(dataframe)
    change_column_place(dataframe)
    value_to_drop = ['אחר']
    dataframe = dataframe.drop(dataframe[dataframe['type'].isin(value_to_drop)].index)
    return dataframe

prepare_data(data)

