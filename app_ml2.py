print('star')

import streamlit as st
import numpy as np
import joblib as jb
import pandas as pd

st.title('modele de prediction finance')
st.header('appli de Seydou Dia')
st.markdown('*appli utilisant model de machine learning finance*')

# # Load the model

model=jb.load(filename='final_model_check2.joblib')

# Récupérer les entrées de l'utilisateur
country =st.number_input('country', value=0)
year =st.number_input('year', value=2018)
location_type=st.number_input('location_type', value=0)
cellphone_access=st.number_input('cellphone_access', value=1)
household_size=st.number_input('household_size', value=3)
age_of_respondent=st.number_input('age_of_respondent', value=24)
gender_of_respondent=st.number_input('gender_of_respondent', value=0)
relationship_with_head=st.number_input('relationship_with_head', value=5)
marital_status=st.number_input('marital_status', value=2)
education_level=st.number_input('education_level', value=3)
job_type=st.number_input('job_type', value=9)

# Créer un DataFrame avec les nouvelles données

new_dat = pd.DataFrame({'country' : [country], 
                        'year' : [year], 
                        'location_type' : [location_type], 
                        'cellphone_access' : [cellphone_access], 
                        'household_size' : [household_size], 
                        'age_of_respondent' : [age_of_respondent], 
                        'gender_of_respondent' : [gender_of_respondent],
                        'relationship_with_head' : [relationship_with_head], 
                        'marital_status' : [marital_status], 
                        'education_level' : [education_level], 
                        'job_type' : [job_type]})

# Faire la prédiction
if st.button('Predic'):
    predir = model.predict(new_dat)

    # Afficher le résultat
    result = "le resultat est  "+str(predir[0])
    st.success(result)

    print('fin')