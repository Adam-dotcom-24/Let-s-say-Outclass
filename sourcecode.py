import streamlit as st
import pandas as pd
import seaborn as sns
import pickle

st.write("# Advertising Sales App")
st.write("This app predicts the sales value!")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 0.0, 400.0 , 0.0)
    Radio = st.sidebar.slider('Radio', 0.0, 400.0, 0.0)
    Newspaper = st.sidebar.slider('Newspaper', 0.0, 200.0, 0.0)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper,}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

modelSales = pickle.load(open("modelsales.h5", "rb"))
new_pred = modelSales.predict(df)

st.subheader('Prediction')
st.write(new_pred)

