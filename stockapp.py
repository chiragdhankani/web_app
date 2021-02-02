#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np
import talib
import yfinance as yf
from datetime import date
import TAcharts
import math
from TAcharts.indicators.ichimoku import Ichimoku


# In[9]:


image = Image.open('download.jpg')
st.image(image, use_column_width=True)


# In[3]:


st.write("""
#  Stock Analysis
""")

st.sidebar.header('User Input Features')


# In[10]:


today = date.today()


# In[22]:


df= pd.read_csv("NiftyAdmin.csv")


# In[141]:


sorted_unique_company = df[['SYMBOL']]
sorted_unique_company['SYMBOL'] = sorted_unique_company['SYMBOL']+ '.NS'
symbol = sorted_unique_company['SYMBOL'].to_list()
symbol1 = pd.DataFrame(symbol,columns=['Co'])
Companies = symbol1.Co.unique()
selected_company = st.sidebar.selectbox("Company", Companies)


# In[ ]:




company = pd.read_csv("NiftyAdmin.csv")
df['Symbol'] = df['SYMBOL'] + ".NS"
symbol = df['Symbol'].to_list()
# In[159]:


def load_data(Company):
    data = yf.download(tickers = selected_company , start= "2000-01-01", end= today)
    x= data.copy(deep= True)
#    x = data1.stack()
    x.reset_index(level=0, inplace=True)
#    x['weekday'] = x['Date'].dt.day_name()
#    x.reset_index(level=0, inplace=True)
#    x = x.rename(columns={'index': 'Company'})
#    x = x.sort_values(by=['Company','Date'], axis=0, ascending=True, ignore_index=True)
    return x
        


# In[160]:


x = load_data(selected_company)


# In[161]:


x['change'] = (x['Close']- x['Close'].shift(+1));
x['change%'] = (x['Close']- x['Close'].shift(+1))/x['Close'].shift(+1) * 100; 
x['pivot'] = (x['High']+x['Low']+x['Close'])/3;
x['ATR_14'] = talib.ATR(x['High'], x['Low'], x['Close'], timeperiod=14);
x['Daily H-L'] = (x['High']-x['Low']);
x['Daily (H-L)%'] =  x['Daily H-L']/x['Close'].shift(+1)*100;
x['H-L%_Avg'] = x['Daily (H-L)%'].rolling(10).mean();
x['H-L%_STD'] = x['Daily (H-L)%'].rolling(10).std();
x['20D_change%'] = x['change%'].rolling(20).mean()
x['U/D'] = np.where(x['change']>0, 1,0)
x['10_H_Avg'] = x['High'].rolling(10).mean();
x['10_L_Avg'] = x['Low'].rolling(10).mean();
x['10_Hmax'] = x['High'].rolling(10).max();
x['10_Lmin'] = x['Low'].rolling(10).min();
x['SAP_Pivot_+-50']= np.where(((x['Date'] > '2020-01-01') & (x['Date'] <= '2021-01-21')),
                     (x['Open']+(x['High'].rolling(10).max())+ (x['Low'].rolling(10).min())+x['Close'])/4, np.nan);


# In[162]:


mask1= np.where(x['Close']>x['SAP_Pivot_+-50'],('1'),('0'));
x['1/0 SAP Pivot']= np.where((x['Date'] > '2020-01-01') & (x['Date'] <= '2021-01-21'), mask1, np.nan);
macd1, macdsignal1, macdhist1 = talib.MACD(x['Close'], fastperiod=10, slowperiod=20, signalperiod=9)
x['20_P'] = (x['Close']-x['Close'].rolling(20).mean())
x['macd10-20']= macd1
x['macdsignal10-20']= macdsignal1
x['macdhist_10-20']= macdhist1
x['P_Abv_20H_MA'] = (x['Close'] - x['High'].rolling(20).mean());
x['P_Bel_20L_MA'] = (x['Close'] - x['Low'].rolling(20).mean());
x['10-20 Cross%'] = ((x['Close'].rolling(10).mean()-x['Close'].rolling(20).mean())/x['Close'].rolling(20).mean())*100;
x['20_P%MA'] = ((x['20_P']/x['Close'])*100);
data = np.where(x['20_P%MA']>0, 1,0)
data1 = data
s = pd.Series(data1)    
data1 = s.groupby(s.eq(0).cumsum()).cumsum().tolist()
x['20_P%MA_Days+']= data1
data2 = np.where(x['20_P%MA']<0, 1,0)
data3 = data2
p = pd.Series(data3)    
data3 = p.groupby(p.eq(0).cumsum()).cumsum().tolist()
x['20_P%MA_Days-']= data3
rsi = talib.RSI(x['Close'], timeperiod=15)
x['15_RSI'] =rsi


# In[163]:


macd2, macdsignal2, macdhist2 = talib.MACD(x['Close'], fastperiod=20, slowperiod=40, signalperiod=9)
x['40_P'] = (x['Close']-x['Close'].rolling(40).mean())
x['macd20-40']= macd2
x['macdsignal20-40']= macdsignal2
x['macdhist_20-40']= macdhist2
x['P_Abv_40H_MA'] = (x['Close'] - x['High'].rolling(40).mean());
x['P_Bel_40L_MA'] = (x['Close'] - x['Low'].rolling(40).mean());
x['20-40 Cross%'] = ((x['Close'].rolling(20).mean()-x['Close'].rolling(40).mean())/x['Close'].rolling(40).mean())*100;
x['40_P%MA'] = ((x['40_P']/x['Close'])*100);
data5 = np.where(x['40_P%MA']>0, 1,0)
data6 = data5
q = pd.Series(data6)    
data6 = q.groupby(q.eq(0).cumsum()).cumsum().tolist()
x['40_P%MA_Days+']= data6
data7 = np.where(x['40_P%MA']<0, 1,0)
data8 = data7
r = pd.Series(data8)    
data8 = r.groupby(r.eq(0).cumsum()).cumsum().tolist()
x['40_P%MA_Days-']= data8
x['10 Conversion line'] =(x['High'].rolling(10).max()+x['Low'].rolling(10).min())/2;
x['Base line 20D HL'] =(x['High'].rolling(20).max()+x['Low'].rolling(20).min())/2;
x['Future fast line'] =  (x['10 Conversion line'] + x['Base line 20D HL'])/2;
x['Future slow line'] = (x['High'].rolling(40).max()+x['Low'].rolling(40).min())/2;
x['20 D old Close'] = np.where(x['Future fast line']>0 ,x['Close'].shift(+20), np.nan);  
x['40 D old Close'] = np.where(x['Future fast line']>0 ,x['Close'].shift(+40), np.nan); 


# In[164]:


def roundup(t):
    return int(math.ceil(t / 10.0)) * 10
f1 = np.vectorize(roundup)
a1 = (x['Close'].rolling(10).max())-1
a2 = np.where(a1.isnull(),a1.fillna(0), a1)
x['10 D Max'] = f1(a2)
def roundown(d):
    return int(math.floor(d / 10.0)) * 10
f2 = np.vectorize(roundown)
a3 = (x['Close'].rolling(10).min())-1
a4 = np.where(a3.isnull(),a3.fillna(0), a3)
x['10 D Min'] = f2(a4)

upperband, middleband, lowerband = talib.BBANDS(x['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

x['Upper BB'] = upperband
x['Lower BB'] = lowerband
x['Middle BB'] = middleband


# In[ ]:





# In[ ]:





# In[165]:


df_final = x# .loc[data1.Company == selected_company]
df_final.fillna(0)


# In[166]:


st.dataframe(df_final)


# In[167]:


st.markdown("""
            
## All Features:

These Are total 57 Columns with various analysis on Data of 522 Companies,  

- **Company Name**
- **Date**
- **Adjusted close Price**
- **Close:** Closing price on that date
- **High:** Highest price of stock on that date
- **Low:** Lowest price of stock on that day
- **Open:** Opening Price of stoc on that day
- **Volume:** No. of stock traded on that day
- **Weekday:** The Day of week
- **Change:** Change in Today'd Closing price of stock from yesterday's Closing
- **Change:** % of change in closing price of the stock
- **pivot:** Average of Open, High and Low prices of stock
- **ATR_14:** 14 Day Average True Range of the stock
- **Daily H-L:** Daily difference of Highest and Lowest Price of stock
- **Daily (H-L)%:** Percentage Difference between H&L with previous day Closing
- **H-L%_Avg:** 10 day Average of % difference between H&L 
- **H-l%_std:** 10 day Standard Deviation of % difference between H&L
- **20D_change%:** 20 Day average of change% of Closing Price 
- **U/D:** Where Price Change is positive=1, For Price change Negative=0
- **10_H_Avg** 10 Day Average of High
- **10_L_Avg:** 10 Day Average of Low 
- **10_Hmax:** maximum of High in 10 Days
- **10_Lmin:** minimun of Low in 10 Days
- **SAP_Pivot_+-50:** Avereage of Today's Open & Close, And 10 days Maximum High & Minimum Low
- **1/0 SAP Pivot:** When 'Closing Price' is greater than **SAP_Pivot_+-50** give 1 else 0  
- **20_p:** Today's Closing subracted by 20 Day Average Closing
- **macd10-20:** MACD(Moving Average Convergence & Diveregence) line for 10-20 Days
- **macdsignal10-20:** MACD signal line for 10-20 Days
- **macdhist_10-20:** MACD Histogram for 10-20 Days
- **P_Abv_20H_MA:** Difference between Today's Closing Price & 20D Average of High Price
- **P_Bel_20L_MA:** Difference between Today's Closing Price & 20D Average of Low Price
- **10-20 Cross%:** Percent difference between 10 Day Avg. Closing & 20 Day Avg. Closing
- **20_P%MA:** percent change of 20_p by today's Closing 
- **20_P%AM_Days+:** Cummlative of Days when 20_P%MA is greater than 0
- **20_P%MA_Days-:** Cummlative of Days when 20_P%MA is greater than 0
- **15_RSI:** 15 Day RSI (Relative Stregth Index)
- **40_p:** Today's Closing subracted by 40 Day Average Closing
- **macd20-40:** MACD(Moving Average Convergence & Diveregence) line for 20-40 Days
- **macdsignal20-40:** MACD signal line for 20-40 Days
- **macdhist_20-40:** MACD Histogram for 20-40 Days
- **P_Abv_40H_MA:** Difference between Today's Closing Price & 40D Average of High Price
- **P_Bel_40L_MA:** Difference between Today's Closing Price & 40D Average of Low Price
- **20-40 Cross%:** Percent difference between 20 Day Avg. Closing & 40 Day Avg. Closing
- **40_P%MA:** percent change of 40_p by today's Closing 
- **40_P%AM_Days+:** Cummlative of Days when 40_P%MA is greater than 0
- **40_P%MA_Days-:** Cummlative of Days when 40_P%MA is greater than 0
- **10 Conversion line:** Avg. of 10 Day Maximum High & Minimum Low 
- **Base line 20D HL:** Avg. of 20 Day Maximum High & Minimum Low
- **Future fast line:** Average of 10 D Conversion & 20 D Base line
- **Future slow line: (D)** Avg. of 40 Day Maximum High & Minimum Low
- **20 D old Close:** 20 Days old Closing Price
- **40 D old Close:** 40 Days old Closing Price 
- **10 D Max:** Max Closing price in 10 Days Roundup to nearest tens 
- **10 D Min:** Min Closing price in 10 Days Rounddown to nearest tens
- **upper BB:** Bollinger Band, Upper Band with 2 STD
- **Lower BB:** Bollinger Band, Lower Band with 2 STD
- **Middle BB:** 20 Days Simple moving Average            
            
""")


# In[ ]:




