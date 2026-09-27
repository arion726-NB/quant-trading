# coding: utf-8

# In[1]:

#this is to London, the greatest city in the world
#i was a Londoner, proud of being Londoner, how i love the city!
#to St Paul, Tate Modern, Millennium Bridge and so much more!

#okay, lets get down to business
#the idea of london break out strategy is to take advantage of fx trading hour
#basically fx trading is 24 hour non stop for weekdays
#u got tokyo, before tokyo closes, u got london
#in the afternoon, u got new york, when new york closes, its sydney
#and several hours later, tokyo starts again
#however, among these three major players
#london is where the majority trades are executed
#not sure if it will stay the same after brexit actually takes place
#what we intend to do is look at the last trading hour before london starts
#we set up our thresholds based on that hours high and low
#when london market opens, we examine the first 30 minutes
#if it goes way above or below thresholds
#we long or short certain currency pairs
#and we clear our positions based on target and stop loss we set
#if they havent reach the trigger condition by the end of trading hour
#we exit our trades and close all positions

#it sounds easy in practise
#just a simple prediction of london fx market based on tokyo market
#but the code of london breakout is extremely time consuming
#first, we need to get one minute frequency dataset for backtest
#i would recommend this website
# http://www.histdata.com/download-free-forex-data/?/excel/1-minute-bar-quotes
#we can get as many as free datasets of various currency pairs we want
#before our backtesting, we should cleanse the raw data
#what we get from the website is one minute frequency bid-ask price
#i take the average of em and add a header called price
#i save it on local disk then read it via python
#please note that this website uses new york time zone utc -5
#for non summer daylight saving time
#london market starts at gmt 8 am
#which is est 3 am
#daylight saving time is another story
#what a stupid idea it is
#yahoo finance update
#the original read gbpusd minute data in est from histdata.com
#now 5 minute bars come from yahoo finance in the exchange's local time
#fx pairs such as GBPUSD=X are in uk time, so the tokyo hour is 7am to 8am
#london opens at 8am and closes at 5pm
#stocks have no tokyo hour before their open
#for 2330.TW and other stocks we use the first 30 minutes of the day
#as the opening range, then trade the breakout until the last bar
#this is the classic opening range breakout
import matplotlib.pyplot as plt
import pandas as pd
import yahoo_data

# In[2]:

def london_breakout(df):
    
    df['signals']=0

    #cumsum is the cumulated sum of signals
    #later we would use it to control our positions
    df['cumsum']=0

    #upper and lower are our thresholds
    df['upper']=0.0
    df['lower']=0.0

    return df


#split one day into the range window and the trading session
def daily_windows(bars,fx,open_minutes):

    if fx:
        minutes=bars.index.hour*60+bars.index.minute
        tokyo=bars[(minutes>=7*60) & (minutes<8*60)]
        london=bars[(minutes>=8*60) & (minutes<17*60)]
        return tokyo,london

    market_open=bars.index[0]+pd.Timedelta(minutes=open_minutes)
    return bars[bars.index<market_open],bars[bars.index>=market_open]


def signal_generation(df,method,column='Close',fx=True):
    
    #risky_stop is the risk tolerance
    #the original used 100 basis points on gbpusd around 1.32
    #we express it relative to the price so it fits stocks too
    #open_minutes is the length of the window to trigger a trade
    #and the length of the opening range for stocks
    risky_stop=0.0075
    open_minutes=30
    
    signals=method(df)
    
    for day,bars in signals.groupby(signals.index.normalize()):
        
        #the last trading hour of tokyo, or the opening range for stocks
        #we use max, min to define the real threshold
        tokyo,london=daily_windows(bars,fx,open_minutes)
        if len(tokyo)==0 or len(london)==0:
            continue
        
        upper=max(tokyo[column])
        lower=min(tokyo[column])
        stop=risky_stop*upper
        entry_end=london.index[0]+pd.Timedelta(minutes=open_minutes)
        
        position=0
        executed_price=float(0)
        
        for i in london.index:
            
            price=signals.at[i,column]
            
            #when its market close, we clear any position left open
            #if there is no open position, -0 is still 0
            if i==london.index[-1]:
                signals.at[i,'signals']=-position
                position=0
            
            #the first 30 minutes after the market opens
            elif i<entry_end:
                signals.at[i,'upper']=upper
                signals.at[i,'lower']=lower
                
                #only the first price above upper threshold triggers the signal
                #also, if it goes skyrocketing beyond our risk tolerance
                #we set it as a false alarm
                #we also need to store the price when we execute a trade
                #its for stop loss calculation
                if price-upper>0 and price-upper<=stop and position<1:
                    signals.at[i,'signals']=1
                    position+=1
                    executed_price=price
                
                #vice versa
                if price-lower<0 and lower-price<=stop and position>-1:
                    signals.at[i,'signals']=-1
                    position-=1
                    executed_price=price
            
            #during trading hour after opening but before closing
            #if there is any open position
            #we set our condition at original executed price +/- half of the risk interval
            #when it goes above or below our risk tolerance
            #we clear positions to claim profit or loss
            elif position!=0:
                if price>executed_price+stop/2 or price<executed_price-stop/2:
                    signals.at[i,'signals']=-position
                    position=0
    
    signals['cumsum']=signals['signals'].cumsum()
    
    return signals


# In[3]:

def plot(new,column='Close'):
    
    #pick the latest day we execute a trade
    traded=new.index[new['signals']!=0].normalize().unique()
    date=traded[-1] if len(traded)>0 else new.index[-1].normalize()
    day=new[new.index.normalize()==date]
    
    #the first plot is the actual trading day
    fig=plt.figure()
    ax=fig.add_subplot(111)
    day[column].plot(label='price')
    ax.plot(day.loc[day['signals']>0].index,day[column][day['signals']>0],lw=0,marker='^',c='g',label='LONG')
    ax.plot(day.loc[day['signals']<0].index,day[column][day['signals']<0],lw=0,marker='v',c='r',label='SHORT')
    
    #mark the window where trades can be triggered
    window=day.index[day['upper']!=0]
    if len(window)>0:
        plt.axvline(window[0],linestyle=':',c='k')
        plt.axvline(window[-1],linestyle=':',c='k')
    
    plt.legend(loc='best')
    plt.title('London Breakout %s'%date.strftime('%Y-%m-%d'))
    plt.ylabel('price')
    plt.xlabel('Date')
    plt.grid(True)
    plt.show()
    
    #the second plot zooms in the market opening
    if len(window)==0:
        return
    news=day[(day.index>=window[0]-pd.Timedelta(minutes=30)) & \
             (day.index<=window[-1]+pd.Timedelta(minutes=30))]
    
    fig=plt.figure()
    bx=fig.add_subplot(111)
    bx.plot(news.loc[news['signals']>0].index,news[column][news['signals']>0],lw=0,marker='^',markersize=10,c='g',label='LONG')
    bx.plot(news.loc[news['signals']<0].index,news[column][news['signals']<0],lw=0,marker='v',markersize=10,c='r',label='SHORT')
    bx.plot(news.loc[news['upper']!=0].index,news['upper'][news['upper']!=0],lw=0,marker='.',markersize=7,c='#BC8F8F',label='upper threshold')
    bx.plot(news.loc[news['lower']!=0].index,news['lower'][news['lower']!=0],lw=0,marker='.',markersize=5,c='#FF4500',label='lower threshold')
    bx.plot(news[column],label='price')
    plt.grid(True)
    plt.ylabel('price')
    plt.xlabel('time interval')
    plt.xticks([])
    plt.title('%s Market Opening'%date.strftime('%Y-%m-%d'))
    plt.legend(loc='best')
    plt.show()


# In[4]:

def main():
    
    #yahoo finance only keeps 5 minute bars for the last 60 days
    #run as python "London Breakout backtest.py" [ticker]
    #taiwan stocks can be given as 2330, 6488, 0050 ...
    ticker,_,_=yahoo_data.cli_args('GBPUSD=X',None,None)
    df=yahoo_data.download_intraday(ticker)
    
    signals=signal_generation(df,london_breakout,fx=yahoo_data.is_fx(ticker))
    plot(signals)

#how to calculate stats could be found from my other code called Heikin-Ashi
# https://github.com/je-suis-tm/quant-trading/blob/master/heikin%20ashi%20backtest.py

if __name__ == '__main__':
    main()
