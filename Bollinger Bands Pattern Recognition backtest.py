
# coding: utf-8

# In[1]:

#bollinger bands is a simple indicator
#just moving average plus moving standard deviation
#but pattern recognition is a differenct case
#visualization is easy for human to identify the pattern
#but for the machines, we gotta find a different approach
#when we talk about pattern recognition these days
#people always respond with machine learning
#why machine learning when u can use arithmetic approach 
#which is much faster and simpler?

#there are many patterns for recognition
#top m, bottom w, head-shoulder top, head-shoulder bottom, elliott waves
#in this content, we only discuss bottom w
#top m is just the reverse of bottom w
#rules of bollinger bands and bottom w can be found in the following link:
# https://www.tradingview.com/wiki/Bollinger_Bands_(BB)

import pandas as pd
import matplotlib.pyplot as plt
import copy
import numpy as np


# In[2]:
import yahoo_data


# In[3]:

#first step is to calculate moving average and moving standard deviation
#we plus/minus two standard deviations on moving average
#we get our upper, mid, lower bands
def bollinger_bands(df):
    
    data=copy.deepcopy(df)
    data['std']=data['price'].rolling(window=20,min_periods=20).std()
    data['mid band']=data['price'].rolling(window=20,min_periods=20).mean()
    data['upper band']=data['mid band']+2*data['std']
    data['lower band']=data['mid band']-2*data['std']
    
    return data


# In[4]:


#the signal generation is a bit tricky
#there are four conditions to satisfy
#for the shape of w, there are five nodes
#from left to right, top to bottom, l,k,j,m,i
#when we generate signals
#the iteration node is the top right node i, condition 4
#first, we find the middle node j, condition 2
#next, we identify the first bottom node k, condition 1
#after that, we point out the first top node l
#l is not any of those four conditions
#we just use it for pattern visualization
#finally, we locate the second bottom node m, condition 3
#plz refer to the following link for my poor visualization
# https://github.com/je-suis-tm/quant-trading/blob/master/preview/bollinger%20bands%20bottom%20w%20pattern.png
def signal_generation(data,method,alpha=0.0001,beta=0.0001):
    
    #according to investopedia
    #for a double bottom pattern
    #we should use 3-month horizon which is 75
    period=75
    
    #alpha denotes the difference between price and bollinger bands
    #if alpha is too small, its unlikely to trigger a signal
    #if alpha is too large, its too easy to trigger a signal
    #which gives us a higher probability to lose money
    #beta denotes the scale of bandwidth
    #when bandwidth is larger than beta, it is expansion period
    #when bandwidth is smaller than beta, it is contraction period
    #both are relative to the price so the same values work for
    #gbpusd around 1.3 and a taiwan stock around 1000
    #the original minute data used 0.0001 on gbpusd, about 0.0075%
    #daily bars move far more, run_all.py and main() pass 1%
    
    df=method(data)
    df['signals']=0
    
    #as usual, cumsum denotes the holding position
    #coordinates store five nodes of w shape
    #later we would use these coordinates to draw a w shape
    df['cumsum']=0
    df['coordinates']=''
    
    for i in range(period,len(df)):
        
        #moveon is a process control
        #if moveon==true, we move on to verify the next condition
        #if false, we move on to the next iteration
        #threshold denotes the value of node k
        #we would use it for the comparison with node m
        #plz refer to condition 3
        moveon=False
        threshold=0.0
        
        #bottom w pattern recognition
        #there is another signal generation method called walking the bands
        #i personally think its too late for following the trend
        #after confirmation of several breakthroughs
        #maybe its good for stop and reverse
        #condition 4
        if (df['price'][i]>df['upper band'][i]) and \
        (df['cumsum'][i]==0):
            
            for j in range(i,i-period,-1):                
                
                #condition 2
                if (np.abs(df['mid band'][j]-df['price'][j])<alpha*df['price'][i]) and \
                (np.abs(df['mid band'][j]-df['upper band'][i])<alpha*df['price'][i]):
                    moveon=True
                    break
            
            if moveon==True:
                moveon=False
                for k in range(j,i-period,-1):
                    
                    #condition 1
                    if (np.abs(df['lower band'][k]-df['price'][k])<alpha*df['price'][i]):
                        threshold=df['price'][k]
                        moveon=True
                        break
                        
            if moveon==True:
                moveon=False
                for l in range(k,i-period,-1):
                    
                    #this one is for plotting w shape
                    if (df['mid band'][l]<df['price'][l]):
                        moveon=True
                        break
                    
            if moveon==True:
                moveon=False        
                for m in range(i,j,-1):
                    
                    #condition 3
                    if (df['price'][m]-df['lower band'][m]<alpha*df['price'][i]) and \
                    (df['price'][m]>df['lower band'][m]) and \
                    (df['price'][m]<threshold):
                        df.at[i,'signals']=1
                        df.at[i,'coordinates']='%s,%s,%s,%s,%s'%(l,k,j,m,i)
                        df['cumsum']=df['signals'].cumsum()
                        moveon=True
                        break
        
        #clear our positions when there is contraction on bollinger bands
        #contraction on the bandwidth is easy to understand
        #when price momentum exists, the price would move dramatically for either direction
        #which greatly increases the standard deviation
        #when the momentum vanishes, we clear our positions
        
        #note that we put moveon in the condition
        #just in case our signal generation time is contraction period
        #but we dont wanna clear positions right now
        if (df['cumsum'][i]!=0) and \
        (df['std'][i]<beta*df['price'][i]) and \
        (moveon==False):
            df.at[i,'signals']=-1
            df['cumsum']=df['signals'].cumsum()
            
    return df


# In[5]:

#visualization
def plot(new):
    
    #as usual we could cut the dataframe into a small slice
    #for a tight and neat figure
    #a and b denotes entry and exit of a trade
    trades=list(new[new['signals']!=0].iloc[:2].index)
    if len(trades)<2 or new['signals'].loc[trades[0]]!=1:
        print('no complete bollinger bands trade in this period, nothing to plot')
        return
    a,b=trades
    
    newbie=new[a-85:b+30]
    newbie.set_index(pd.to_datetime(newbie['date']),inplace=True)

   
    fig=plt.figure(figsize=(10,5))
    ax=fig.add_subplot(111)
    
    #plotting positions on price series and bollinger bands
    ax.plot(newbie['price'],label='price')
    ax.fill_between(newbie.index,newbie['lower band'],newbie['upper band'],alpha=0.2,color='#45ADA8')
    ax.plot(newbie['mid band'],linestyle='--',label='moving average',c='#132226')
    ax.plot(newbie['price'][newbie['signals']==1],marker='^',markersize=12, \
            lw=0,c='g',label='LONG')
    ax.plot(newbie['price'][newbie['signals']==-1],marker='v',markersize=12, \
            lw=0,c='r',label='SHORT')
    
    #plotting w shape
    #we locate the coordinates then find the exact date as index
    temp=newbie['coordinates'][newbie['signals']==1]
    indexlist=list(map(int,temp[temp.index[0]].split(',')))
    ax.plot(newbie['price'][pd.to_datetime(new['date'].iloc[indexlist])], \
            lw=5,alpha=0.7,c='#FE4365',label='double bottom pattern')
    
    #add some captions
    plt.text((newbie.loc[newbie['signals']==1].index[0]), \
             newbie['lower band'][newbie['signals']==1],'Expansion',fontsize=15,color='#563838')
    plt.text((newbie.loc[newbie['signals']==-1].index[0]), \
             newbie['lower band'][newbie['signals']==-1],'Contraction',fontsize=15,color='#563838')
    
    plt.legend(loc='best')
    plt.title('Bollinger Bands Pattern Recognition')
    plt.ylabel('price')
    plt.grid(True)
    plt.show()


# In[6]:

#ta-da
def main():
    
    #the original used gbpusd minute data from histdata.com
    #now daily bars come from yahoo finance
    #run as python "Bollinger Bands Pattern Recognition backtest.py" [ticker] [start] [end]
    #taiwan stocks can be given as 2330, 6488, 0050 ...
    ticker,stdate,eddate=yahoo_data.cli_args('GBPUSD=X','2016-01-01','2024-12-31')
    data=yahoo_data.download(ticker,start=stdate,end=eddate)
    df=pd.DataFrame({'date':data.index,'price':data['Close'].values})
    
    signals=signal_generation(df,bollinger_bands,alpha=0.02,beta=0.015)

    new=copy.deepcopy(signals)
    plot(new)

#how to calculate stats could be found from my other code called Heikin-Ashi
# https://github.com/je-suis-tm/quant-trading/blob/master/heikin%20ashi%20backtest.py


if __name__ == '__main__':
    main()
