import json
import openai
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import requests
from textblob import TextBlob
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from textbase import bot, Message
from textbase.models import OpenAI
from typing import List


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

def get_news_articles(stock_ticker):
    sources = [
        ("https://www.moneycontrol.com/news/business/stocks/", "moneycontrol"),
        ("https://www.bloomberg.com/quote/" + stock_ticker + ":IN", "bloomberg"),
        ("https://www.reuters.com/finance/stocks/overview/" + stock_ticker, "reuters"),
        ("https://www.cnbc.com/quotes/" + stock_ticker, "cnbc")
    ]
    
    all_news_data = []
    
    for source, source_type in sources:
        response = requests.get(source)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            
            if source_type == "moneycontrol":
                articles = soup.find_all("li", class_="clearfix")
            elif source_type == "bloomberg":
                articles = soup.find_all("div", class_="story-story-grid-story__3MlKt")
            elif source_type == "reuters":
                articles = soup.find_all("div", class_="topStory")
            elif source_type == "cnbc":
                articles = soup.find_all("div", class_="QuotePage-summary")
            
            for article in articles:
                text = article.get_text() if source_type == "moneycontrol" else article.find("h1").get_text()
                all_news_data.append({"source": source_type, "headline": text})
    
    return all_news_data

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return sentiment

def get_final_sentiment(sentiments):
    positive_count = sum(1 for sentiment in sentiments if sentiment > 0)
    negative_count = sum(1 for sentiment in sentiments if sentiment < 0)
    
    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "Neutral"
def get_stock_sentiment(stock_ticker):
    news_data = get_news_articles(stock_ticker)
    
    if news_data:
        sentiments = [analyze_sentiment(news["headline"]) for news in news_data]
        final_sentiment = get_final_sentiment(sentiments)
        
        result = {
            "ticker": stock_ticker,
            "sentiment": final_sentiment,
            "news": [{"source": news['source'], "headline": news['headline']} for news in news_data]
        }
    else:
        result = {"error": "Error fetching news articles."}

    return result
def sentiment_of_stock(company):
    stock_ticker = company  # Assuming the user provides a stock ticker as input
    sentiment_result = get_stock_sentiment(stock_ticker)
    
    response = f"Sentiment for {sentiment_result['ticker']}: {sentiment_result['sentiment']}\n"
    for news in sentiment_result.get('news', []):
        response += f"Source: {news['source']}\n"
        response += f"Headline: {news['headline']}\n"
        response += f"Sentiment: {analyze_sentiment(news['headline'])}\n"
        response += "-" * 50 + "\n"
    
    return response
    
functions=[
    
    {
        'name':'get_stock_price',
        'description':'Get the current stock price of a company',
        'parameters':{
            'type':'object',
            'properties':{
                'company':{
                    'type':'string',
                    'description':'The company to get the stock price of'
                }
            },
            'required':['company']
        }
    },
    {
        'name':'stock_info',
        'description':'Get the stock info of a company',
        'parameters':{
            'type':'object',
            'properties':{
                'company':{
                    'type':'string',
                    'description':'The company to get the stock info of'
                }
            },
            'required':['company']
        }
    },
    {
        'name':'calculate_SMA',
        'description':'Calculate the Simple Moving Average of a company',
        'parameters':{
            'type':'object',
            'properties':{
                'company':{
                    'type':'string',
                    'description':'The company to calculate the SMA of'
                },
                'window':{
                    'type':'integer',
                    'description':'The window to calculate the SMA over'
                }
            },
            'required':['company','window']
        }
    },
    {
        'name':'calculate_EMA',
        'description':'Calculate the Exponential Moving Average of a company',
        'parameters':{
            'type':'object',
            'properties':{
                'company':{
                    'type':'string',
                    'description':'The company to calculate the EMA of'
                },
                'window':{
                    'type':'integer',
                    'description':'The window to calculate the EMA over'
                }
            },
            'required':['company','window']
        }
    },
    {
        'name':'calculate_MACD',
        'description':'Calculate the Moving Average Convergence Divergence of a company',
        'parameters':{
            'type':'object',
            'properties':{
                'company':{
                    'type':'string',
                    'description':'The company to calculate the MACD of'
                }
            },
            'required':['company']
        }
    },
    {
        'name':'calculate_RSI',
        'description':'Calculate the Relative Strength Index of a company',
        'parameters':{
            'type':'object',
            'properties':{
                'company':{
                    'type':'string',
                    'description':'The company to calculate the RSI of'
                },
                
            },
            'required':['company']
        }
    },
    {
        'name':'plot_stock_price',
        'description':'Plot the stock price of a company over the last year',
        'parameters':{
            'type':'object',
            'properties':{
                'company':{
                    'type':'string',
                    'description':'The company to plot the stock price of'
                }
            },
            'required':['company']
        }
    },
    {
        'name':'sentiment_of_stock',
        'description':'Get the sentiment of a stock',
        'parameters':{
            'type':'object',
            'properties':{
                'company':{
                    'type':'string',
                    'description':'The company to get the sentiment of'
                }
            },
            'required':['company']
        }
    }
    

]

available_functions = {
    'get_stock_price':get_stock_price,
    'stock_info':stock_info,
    'calculate_SMA':calculate_SMA,
    'calculate_EMA':calculate_EMA,
    'calculate_MACD':calculate_MACD,
    'calculate_RSI':calculate_RSI,
    'sentiment_of_stock':sentiment_of_stock

}
@bot()
def financial_advisor(message_history: List[Message], state: dict = None):
   
        if message_history[-1].get('content'):
            response = openai.generate(
                model='gpt-3.5-turbo-16k',
                messages=message_history,
                message_history=message_history,
                max_tokens=2000,
                functions=functions,
                function_call='auto',

            )
            
            response_message = response['choices'][0]['message']
            # Handle the response as in your original code
            if response_message.get('function_call'):
                args_dict={}
                function_name=response_message['function_call']['name']
                function_args=json.loads(response_message['function_call']['arguments'])
                if function_name in ['get_stock_price','calculate_MACD','calculate_RSI','plot_stock_price','stock_info','sentiment_of_stock']:
                    args_dict={'company':function_args.get('company')}
                elif function_name in ['calculate_SMA','calculate_EMA']:
                    args_dict={'company':function_args.get('company'),'window':function_args.get('window')}
                
                function_to_call=available_functions[function_name]
                function_response=function_to_call(**args_dict)
                # if function_name=='plot_stock_price':
                #     return stock_price.png
                # else:
                response = {
                    "data": {
                    "messages": [
                    {
                        "data_type": "STRING",
                        "value": function_response
                    }
                ],
                "state": state
            },
            "errors": [
                {
                    "message": ""
                }
            ]
        }

        return {
            "status_code": 200,
            "response": response
        }
