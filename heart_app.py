# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:46:59 2022

@author: User
"""

import os
import pickle
import numpy as np
import streamlit as st

MODEL_PATH = os.path.join(os.getcwd(),'best_model.pkl')

with open(MODEL_PATH,'rb') as file:
    model = pickle.load(file)

#%%

# The features are
# ['age','cp','trtbps','chol','thalachh','oldpeak','thall']

with st.form("Patient's info"):
    st.write("This form is to predict if a person has heart attack")
    age = st.number_input('Key in your age')
    cp = int(st.radio("Select your chest pain type. 0 for Typical Angina,\
                      1 for Atypical Angina, 2 for Non-Anginal pain\
                          3 for Asymptomatic",(0,1,2,3)))
    trtbps = st.number_input('Key in your resting blood pressure')
    chol = st.number_input('Key in your cholesterol in mg/dl')
    thalachh = st.number_input('Key in your maximum heart rate achieved')
    oldpeak = st.number_input('Key in your previous peak')
    thall = int(st.radio('Select your thal rate',(0,1,2,3)))
    
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        X_new = [age,cp,trtbps,chol,thalachh,oldpeak,thall]
        
        outcome = model.predict(np.expand_dims(np.array(X_new),axis=0))
        
        outcome_dict = {0:'Less chance of getting heart attack',
                        1:'More chance of getting heart attack'}
        
        st.write(outcome_dict[outcome[0]])





