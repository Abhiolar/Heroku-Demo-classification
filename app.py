import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import json




app = Flask(__name__) #Initialize the flask app
model = pickle.load(open('model.pkl', 'rb')) #loading the trained model
scaler = pickle.load(open('scaler.pkl', 'rb')) #loading the scaler object
ohe = pickle.load(open('ohe.pkl', 'rb')) #loading the encoder object



@app.route('/') #Homepage
def home():
    return render_template('index.html')


@app.route('/predict', methods = ['POST'])
def predict():
    
    '''
    For rendering results on HTML GUI
    '''
    
     
   
 
    init_features = [request.form.values()]

    row = init_features[0]
    transform_data = pd.DataFrame(row).transpose()
    transform_data.columns = ['total_items', 'total_sales', 'discounted_sales',
                     'browsing_duration', 'age', 'household_income', 'loyalty_points','month', 'loyalty_card', 'education', 'marital_status', 'region', 'gender']
    
    non_normal = ['total_items', 'total_sales', 'discounted_sales',
                     'browsing_duration', 'age', 'household_income', 'loyalty_points']
    
    data_cat = transform_data[['month', 'loyalty_card', 'education', 'marital_status', 'region', 'gender']]
    data_cont = transform_data[non_normal].astype(float)


    
    predictor_scaled = pd.DataFrame(scaler.transform(data_cont))
    predictor_encoded =ohe.transform(data_cat)
    columns = ohe.get_feature_names(input_features=data_cat.columns)
    cat_features = pd.DataFrame(predictor_encoded.todense(), columns=columns)
    
    test_data = pd.concat([pd.DataFrame(predictor_scaled), cat_features], axis = 1)
    
    
    prediction = (model.predict_proba(test_data)[:,1] >= 0.47).astype(bool)
    
    
    
    return render_template('index.html', prediction_text='Is this an Incomplete Transaction?: {}'.format(prediction)) # rendering the predicted result

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
