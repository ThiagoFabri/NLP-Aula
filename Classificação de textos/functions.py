"""
Source: https://github.com/soft-nougat/dqw-ivves
@author: TNIKOLIC

"""
import json
import streamlit as st
import pandas as pd
import base64
import streamlit.components.v1 as components
from PIL import Image
import pandas as pd
import pycaret as pyc

import io
from PIL import Image
from pprint import pprint
from zipfile import ZipFile
import os
from os.path import basename

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

def app_section_button(option1, option2, option3, option4):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # current page
        col1.markdown(option1) 
    with col2:
        st.markdown(option2, unsafe_allow_html=True) 
    with col3:
        st.markdown(option3, unsafe_allow_html=True) 
    with col4:
        st.markdown(option4, unsafe_allow_html=True) 
      

def app_meta(icon):

    # Set website details
    st.set_page_config(page_title ="Processamento de Linguagem natural", 
                       page_icon=icon, 
                       layout='centered')
    
    # set sidebar width
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        margin-left: -300px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
    The bg will be static and won't take resolution of device into account.
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# definindo a função que fará a previsão usando os dados que o usuário insere
def prediction_classifier(docs_new):

    docs_new = [docs_new]
    # carregar o modelo treinado
    model = joblib.load('mlp_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    docs = list(docs_new)
    #vectorizer = TfidfVectorizer()   
    X_new_tfidf_vectorize = vectorizer.transform(docs)

    sgd_predicted = model.predict(X_new_tfidf_vectorize)

    categories = ['rec.motorcycles', 'rec.autos']
    twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)

    for doc, category in zip(docs_new, sgd_predicted):
        return 'O texto a seguir..: {}, foi classificado como {}. '.format(doc, twenty_train.target_names[category])  