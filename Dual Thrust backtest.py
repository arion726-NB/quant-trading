# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:22:38 2018
@author: Administrator

"""
# In[1]:

#dual thrust is an opening range breakout strategy
#it is very similar to London Breakout
#please check London Breakout if u have any questions
# https://github.com/je-suis-tm/quant-trading/blob/master/London%20Breakout%20backtest.py
#Initially we set up upper and lower thresholds based on previous days open, close, high and low
#When the market opens and the price exceeds thresholds, we would take long/short positions prior to upper/lower thresholds
#However, there is no stop long/short position in this strategy
#We clear all positions at the end of the day
#rules of dual thrust can be found in the following link
# https://www.quantconnect.com/tutorials/dual-thrust-trading-algorithm/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yahoo_data


# In[2]:


#the original used gbpusd minute data from histdata.com
#with the london session hard coded as est 3am to 12pm
#now the data comes from yahoo finance 5 minute bars
#fx pairs keep the london session, 8am to 5pm uk time
#stocks such as 2330.TW trade from the first to the last bar of each day
def session_mask(df,ticker):

    if yahoo_data.is_fx(ticker):
        minutes=df.index.hour*60+df.index.minute
        return (minutes>=8*60) & (minutes<17*60)

    return np.full(len(df),True)


# In[3]:


#data frequency convertion from minute to intra daily
#as we are doing backtesting, we have already got all the datasets we need
#we can create a table to store all open, close, high and low prices
#and calculate the range before we get to signal generation
#otherwise, we would have to put this part inside the loop
#it would greatly increase the time complexity
#however, in real time trading, we do not have futures price
#we have to store all past information in sql db
#we have to calculate the range from db before the market opens
def min2day(df,column,rg):

    #group the session bars by calendar day
    session=df[df['in session']]
    grouped=session[column].groupby(session.index.normalize())

    intraday=pd.DataFrame({'open':grouped.first(),
                           'close':grouped.last(),
                           'high':grouped.max(),
                           'low':grouped.min()})
    intraday['date']=intraday.index

    #preparation
    intraday['range1']=intraday['high'].rolling(rg).max()-intraday['close'].rolling(rg).min()
    intraday['range2']=intraday['close'].rolling(rg).max()-intraday['low'].rolling(rg).min()
    intraday['range']=np.where(intraday['range1']>intraday['range2'],intraday['range1'],intraday['range2'])

    #the range must only use the previous rg days
    #the original included the current day, whose high and low are unknown at the open
    intraday['range']=intraday['range'].shift(1)

    return intraday


#signal generation
#even replace assignment with pandas.at
#it still takes a while for us to get the result
#any optimization suggestion besides using numpy array?
def signal_generation(df,intraday,param,column,rg):

    #we start backtesting on the first day with a full range
    #cumsum is to control the holding of underlying asset
    #sigup and siglo are the variables to store the upper/lower threshold
    #upper and lower are for the purpose of tracking sigup and siglo
    first_day=intraday['range'].dropna().index[0]
    signals=df[df['in session'] & (df.index.normalize()>=first_day)].copy()
    signals['signals']=0
    signals['cumsum']=0
    signals['upper']=0.0
    signals['lower']=0.0
    sigup=float(0)
    siglo=float(0)

    #market opening and closing bar of every day
    days=signals.groupby(signals.index.normalize())
    opening=set(days.head(1).index)
    closing=set(days.tail(1).index)

    #for traversal on time series
    #first we set up thresholds at the beginning of the session
    #if the price exceeds either threshold
    #we will take long/short positions
    for i in signals.index:

        price=signals.at[i,column]

        #market opening
        #set up thresholds
        if i in opening:
            day_range=intraday.at[i.normalize(),'range']
            sigup=float(param*day_range+price)
            siglo=float(-(1-param)*day_range+price)

        #thresholds got breached
        #signals generating
        if (sigup!=0 and price>sigup):
            signals.at[i,'signals']=1
        if (siglo!=0 and price<siglo):
            signals.at[i,'signals']=-1


        #check if signal has been generated
        #if so, use cumsum to verify that we only generate one signal for each situation
        if signals.at[i,'signals']!=0:
            signals['cumsum']=signals['signals'].cumsum()
            if (signals.at[i,'cumsum']>1 or signals.at[i,'cumsum']<-1):
                signals.at[i,'signals']=0

            #if the price goes from below the lower threshold to above the upper threshold during the day
            #we reverse our positions from short to long
            if (signals.at[i,'cumsum']==0):
                if (price>sigup):
                    signals.at[i,'signals']=2
                if (price<siglo):
                    signals.at[i,'signals']=-2

        #by the end of the session
        #we clear all opening positions
        #the whole part is very similar to London Breakout strategy
        if i in closing:
            sigup,siglo=float(0),float(0)
            signals['cumsum']=signals['signals'].cumsum()
            signals.at[i,'signals']-=signals.at[i,'cumsum']

        #keep track of trigger levels
        signals.at[i,'upper']=sigup
        signals.at[i,'lower']=siglo

    signals['cumsum']=signals['signals'].cumsum()

    return signals

#plotting the positions
def plot(signals,intraday,column):

    #we have to do a lil bit slicing to make sure we can see the plot clearly
    #pick the latest day we execute a trade
    traded=signals.index[signals['signals']!=0].normalize().unique()
    date=traded[-1] if len(traded)>0 else signals.index[-1].normalize()
    signew=signals[signals.index.normalize()==date]

    fig=plt.figure(figsize=(10,5))
    ax=fig.add_subplot(111)

    #mostly the same as other py files
    #the only difference is to create an interval for signal generation
    ax.plot(signew.index,signew[column],label=column)
    ax.fill_between(signew.loc[signew['upper']!=0].index,signew['upper'][signew['upper']!=0],signew['lower'][signew['upper']!=0],alpha=0.2,color='#355c7d')
    ax.plot(signew.loc[signew['signals']>0].index,signew[column][signew['signals']>0],lw=0,marker='^',markersize=10,c='g',label='LONG')
    ax.plot(signew.loc[signew['signals']<0].index,signew[column][signew['signals']<0],lw=0,marker='v',markersize=10,c='r',label='SHORT')

    #change legend text color
    lgd=plt.legend(loc='best').get_texts()
    for text in lgd:
        text.set_color('#6C5B7B')

    #add some captions
    plt.text(signew.index[0],signew['upper'].iloc[0],'Upper Bound',color='#C06C84')
    plt.text(signew.index[0],signew['lower'].iloc[0],'Lower Bound',color='#C06C84')

    plt.ylabel(column)
    plt.xlabel('Date')
    plt.title('Dual Thrust %s'%date.strftime('%Y-%m-%d'))
    plt.grid(True)
    plt.show()



# In[4]:
def main():

    #yahoo finance only keeps 5 minute bars for the last 60 days
    #run as python "Dual Thrust backtest.py" [ticker]
    #taiwan stocks can be given as 2330, 6488, 0050 ...
    ticker,_,_=yahoo_data.cli_args('GBPUSD=X',None,None)
    df=yahoo_data.download_intraday(ticker)
    df['in session']=session_mask(df,ticker)

    #rg is the lags of days
    #param is the parameter of trigger range, it should be smaller than one
    #normally ppl use 0.5 to give long and short 50/50 chance to trigger
    rg=5
    param=0.5
    column='Close'

    intraday=min2day(df,column,rg)
    signals=signal_generation(df,intraday,param,column,rg)
    plot(signals,intraday,column)

#how to calculate stats could be found from my other code called Heikin-Ashi
# https://github.com/je-suis-tm/quant-trading/blob/master/heikin%20ashi%20backtest.py

if __name__ == '__main__':
    main()
