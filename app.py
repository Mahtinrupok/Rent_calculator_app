# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
# 2. Create the app object
app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_AusRents (data:AusRents):
    data = data.dict()
    Beds=data['Beds']
    Baths=data['Baths']
    Cars=data['Cars']
    RentYear=data['RentYear']
    Suburb=data['Suburb']
    State=data['State']
    Type=data['Type']
    RentMonth=data['RentMonth']




   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict([[Beds,Baths,Cars,RentYear,Suburb,State,Type,RentMonth]])
    
    return {
         prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:7080
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=7080)
    
#uvicorn app:app --reload