# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:39:48 2023

@author: Ofek biton & Shahaf Malka
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:39:48 2023
@author: Ofek biton & Shahaf Malka
"""

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)
    
   
@app.route('/')
def home():
    print(model.feature_names_in_)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    city = request.form.get('City')
    property_type = request.form.get('type')
    hasParking = request.form.get('hasParking')
    hasAirCondition = request.form.get('hasAirCondition')
    handicapFriendly = request.form.get('handicapFriendly')
    hasMamad = request.form.get('hasMamad')
    room_number = request.form.get('room_number')
    area = request.form.get('Area')
    room_number= float(room_number)
    area = float(area)
    
    string_features = ['אילת', 'באר שבע', 'בית שאן', 'בת ים', 'גבעת שמואל', 'דימונה', 'הוד השרון', 'הרצליה', 'זכרון יעקב', 'חולון', 'חיפה', 'יהוד מונוסון', 'ירושלים', 'כפר סבא', 'מודיעין מכבים רעות', 'נהריה', 'נוף הגליל', 'נס ציונה', 'נתניה', 'פתח תקווה', 'צפת', 'קרית ביאליק', 'ראשון לציון', 'רחובות', 'רמת גן', 'רעננה', 'שוהם', 'תל אביב', 'בית פרטי', 'דו משפחתי', 'דירה בבניין', 'דירת גן', 'פנטהאוז']
    
    new_data = pd.DataFrame({
        'City': [city],
        'type': [property_type],
        'room_number': [room_number],
        'Area': [area],
        'hasParking': [hasParking],
        'hasAirCondition' : [hasAirCondition],
        'hasMamad' : [hasMamad],
        'handicapFriendly' : [handicapFriendly]
    })
    
    for feature in string_features:
        new_data[feature] = 0
        
    new_data_enc = preprocessor.transform(new_data)
    predicted_price = model.predict(new_data_enc)[0]
    text_output = f"Predicted Property Value: {predicted_price:.2f}"

    return render_template('index.html', prediction_text =text_output)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port, debug=True)
