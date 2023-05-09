# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 13:52:25 2023

@author: User
"""

import numpy as np
import pickle
import streamlit as st

##loading the saved model
loaded_model=pickle.load(open('C:/Users/User/OneDrive/Desktop/ML_Project/log_reg_model.sav', 'rb'))

#Create a function for prediction

def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray (input_data).astype(float)

    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)


    if(prediction[0]==0):
      return "The person is not diabetic"

    else:
      return "The person is diabetic"
  
    
   
def main():
    
    #giving a title
    st.title('Diabetes Prediction WebApp')
    #getting input data from user
   
    
    Pregnancies= st.text_input("Number of Pregnencies")
    Glucose= st.text_input("Blood Glucose level")
    BloodPressure= st.text_input("Blood Pressure level")
    SkinThickness= st.text_input("Skin Thickness")
    Insulin= st.text_input("Insulin Level")
    BMI= st.text_input("BMI reading")
    DiabetesPedigreeFunction=st.text_input("Diabetes Predigree Function Value")
    Age=st.text_input("Age")
    
    #code for prediction
    diagnosis=""
    
    #creating a button for prediction
    
    if st.button("Diabetes test Result"):
        diagnosis= diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
        

if __name__=='__main__':
    main()
    
     