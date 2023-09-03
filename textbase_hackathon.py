import json
import openai
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from textblob import TextBlob
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


openai.api_key=open('API_KEY','r').read()

def get_stock_price(company):
    company=company+'.NS'
    dfdata = yf.Ticker(company).history(period='1y')
    return str(dfdata['Close'].iloc[-1])

def stock_info(company):
    company=company+'.NS'
    data = yf.Ticker(company).info
    return str(data)
def calculate_SMA(company,window):
    company=company+'.NS'
    df = yf.Ticker(company).history(period='1y')
    df['SMA'] = df['Close'].rolling(window=window).mean()
    return str(df['SMA'].iloc[-1])

def calculate_EMA(company,window):
    company=company+'.NS'
    df = yf.Ticker(company).history(period='1y')
    df['EMA'] = df['Close'].ewm(span=window,adjust=False).mean()
    return str(df['EMA'].iloc[-1])

def calculate_MACD(company):
    company=company+'.NS'
    df = yf.Ticker(company).history(period='1y').Close
    short_ema = df.ewm(span=12,adjust=False).mean()
    long_ema = df.ewm(span=26,adjust=False).mean()
    MACD=short_ema-long_ema 
    signal = MACD.ewm(span=9,adjust=False).mean()
    MACD_hist = MACD - signal
    return f'{MACD[-1]},{signal[-1]},{MACD_hist[-1]}'


def calculate_RSI(company):
    company=company+'.NS'
    data = yf.Ticker(company).history(period='1y')
    delta = data['Close'].diff(1)
    up=delta.clip(lower=0)
    down=delta.clip(upper=0).abs()
    ema_up=up.ewm(com=13,adjust=False).mean()
    ema_down=down.ewm(com=13,adjust=False).mean()
    rs=ema_up/ema_down
    rsi=100-(100/(1+rs))
    return str(rsi.iloc[-1])
def plot_stock_price(company):
    company=company+'.NS'
    dfdata = yf.Ticker(company).history(period='1y')
    plt.figure(figsize=(12, 6))
    dfdata['Close'].plot(color='blue', linewidth=2)
    plt.title(f"Stock Price for {company}")
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.grid(True, alpha=0.3)

    # Adding a legend
    plt.legend([f'{company} Stock Price'])

    # Adding a background color
    plt.axhspan(0, dfdata['Close'].max(), facecolor='0.95')

    # Adding annotations
    max_price_date = dfdata['Close'].idxmax()
    max_price = dfdata['Close'].max()
    plt.annotate(f'Max Price: {max_price:.2f} INR\nDate: {max_price_date.date()}',
                 xy=(max_price_date, max_price),
                 xytext=(max_price_date - pd.DateOffset(months=2), max_price * 0.9),
                 arrowprops=dict(facecolor='black', arrowstyle='wedge,tail_width=0.7'),
                 fontsize=10,
                 color='black')

    plt.tight_layout()