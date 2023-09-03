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