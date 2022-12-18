# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 23:49:58 2022

@author: smdhuri
"""

import streamlit as st
from datetime import date
import pandas as pd

#import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2016-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Gold Price Prediction App")

value=("date","price")
#selected_value= st.selectbox("Select dataset for prediction",value)

n_days= st.slider("Days of prediction", 1,30)
period = n_days * 24

@st.cache
def load_data(ticker):
    data = pd.read_csv('Gold_data.csv')
    data.reset_index(inplace=True)
    return data

##data_load_state = st.text("Load data...")
data= load_data(value)
##data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['price']))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting

df_train = data[['date','price']]
df_train = df_train.rename(columns={"date": "ds", "price": "y"})

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())


st.write('forecast data')
fig1=plot_plotly(model,forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = model.plot_components(forecast)
st.write(fig2)