# -*- coding: utf-8 -*-
"""
Run every Yahoo Finance backtest in this repository on one ticker
and compare them with the same performance metrics.

    python run_all.py                      # 2330 TSMC, 2016 to today
    python run_all.py 2454                 # MediaTek
    python run_all.py 0050 2020-01-01      # ETF from 2020
    python run_all.py AAPL 2018-01-01 2024-12-31 --pair NVDA AMD

Outputs go to output/<ticker>/
    summary.csv / summary.md   metrics of every strategy
    returns.png, equity.png    overview charts
    <nn>_<strategy>_<k>.png    each strategy's original charts

Metrics are computed the same way for every strategy:
enter at the close of the signal bar, earn the next bar's return.
Taiwan tickers pay 0.1425% broker fee per side plus securities tax
(0.3% on sells, 0.15% for day trades); other markets are frictionless,
as in the original scripts.
"""

import argparse
import datetime
import importlib.util
import os
import random
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,ROOT)
import yahoo_data

warnings.filterwarnings('ignore')


# In[1]: loading the strategy scripts

def load(filename):

    path=os.path.join(ROOT,filename)
    name=os.path.splitext(os.path.basename(filename))[0].replace(' ','_').replace('-','_')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


class ChartSaver:
    """Turn every plt.show() in the original scripts into a saved png."""

    def __init__(self,folder):
        self.folder=folder
        self.prefix='chart'
        self.count=0

        #pyplot sets attributes on plt.show when the backend loads
        #so load it first and patch with a plain function
        plt.close(plt.figure())
        def show(*args,**kwargs):
            self.show(*args,**kwargs)
        plt.show=show

    def start(self,number,name):
        self.prefix='%02d_%s'%(number,name.replace(' ','_'))
        self.count=0

    def show(self,*args,**kwargs):
        for num in plt.get_fignums():
            self.count+=1
            fig=plt.figure(num)
            fig.savefig(os.path.join(self.folder,'%s_%d.png'%(self.prefix,self.count)),
                        dpi=110,bbox_inches='tight')
        plt.close('all')


# In[2]: common performance metrics

TW_FEE=0.001425

#daily strategies whose signals depend on the high/low of the bar
HIGH_LOW_STRATEGIES={'Awesome Oscillator','Heikin-Ashi','Parabolic SAR','Shooting Star'}


def cost_per_side(symbol,intraday):

    if not yahoo_data.is_taiwan(symbol):
        return 0.0

    #securities tax is charged on sells, spread evenly across both sides
    tax=0.0015 if intraday else 0.003

    return TW_FEE+tax/2


def trade_segments(exposure):

    #a trade is a run of bars holding a position in the same direction
    sign=np.sign(exposure.values)
    segments=[]
    start=None
    for i in range(len(sign)):
        if start is not None and sign[i]!=sign[start]:
            segments.append((start,i))
            start=None
        if start is None and sign[i]!=0:
            start=i
    if start is not None:
        segments.append((start,len(sign)))

    return segments


def evaluate(name,freq,price,exposure,cost,note='',leg2=None):
    """
    price, exposure: aligned series, exposure between -1 and 1
    leg2: optional (price, exposure) of a second asset for pair trading
    """

    exposure=exposure.astype(float).fillna(0)
    held=exposure.shift(1).fillna(0)
    strat=held*price.pct_change().fillna(0)-exposure.diff().abs().fillna(exposure.abs())*cost

    if leg2 is not None:
        price2,exposure2=leg2
        exposure2=exposure2.astype(float).fillna(0)
        held2=exposure2.shift(1).fillna(0)
        strat2=held2*price2.pct_change().fillna(0)-exposure2.diff().abs().fillna(exposure2.abs())*cost
        strat=(strat+strat2)/2

    equity=(1+strat).cumprod()

    #intraday bars are compounded into daily returns for sharpe
    daily=(1+strat).groupby(strat.index.normalize()).prod()-1
    years=(strat.index[-1]-strat.index[0]).days/365.25

    trades=trade_segments(exposure)
    #a position opened at the close of bar s and closed at bar e
    #earns from bar s+1 to bar e, entry cost is booked on bar s
    wins=[equity.iloc[min(e,len(equity)-1)]/(equity.iloc[s-1] if s>0 else 1)>1
          for s,e in trades]

    std=daily.std()
    return {
        'strategy':name,
        'frequency':freq,
        'start':strat.index[0].strftime('%Y-%m-%d'),
        'end':strat.index[-1].strftime('%Y-%m-%d'),
        'total return':equity.iloc[-1]-1,
        'annual return':equity.iloc[-1]**(1/years)-1 if years>=1 else np.nan,
        'buy and hold':price.iloc[-1]/price.iloc[0]-1,
        'sharpe':daily.mean()/std*np.sqrt(252) if std>0 else np.nan,
        'max drawdown':(equity/equity.cummax()-1).min(),
        'trades':len(trades),
        'win rate':np.mean(wins) if len(wins)>0 else np.nan,
        'time in market':(held!=0).mean(),
        'note':note,
        '_equity':equity,
    }


# In[3]: the strategies

def run_macd(df,symbol,charts):

    m=load('MACD Oscillator backtest.py')
    m.ma1,m.ma2=10,21
    sig=m.signal_generation(df.copy(),m.macd)
    m.plot(sig.iloc[-250:],symbol)

    return sig['Close'],sig['positions'],'SMA 10/21, long only'


def run_awesome(df,symbol,charts):

    a=load('Awesome Oscillator backtest.py')
    sig=a.signal_generation(df.copy(),a.ewmacd,5,34)
    sig=a.awesome_signal_generation(sig,a.awesome_ma)
    sig.set_index('Date',inplace=True)
    a.plot(sig.iloc[-250:],symbol)

    return sig['Close'],sig['cumsum'].clip(0,1),'5/34 on median price, saucer entries, long only'


def run_heikin_ashi(df,symbol,charts):

    h=load('Heikin-Ashi backtest.py')
    stls=3
    sig=h.signal_generation(df.copy(),h.heikin_ashi,stls)
    sig.set_index('Date',inplace=True)
    h.plot(sig.iloc[-250:].reset_index(),symbol)

    #up to three stacked longs, one third of capital each
    return sig['Close'],sig['signals'].cumsum().clip(0,stls)/stls,'up to 3 stacked longs'


def run_parabolic_sar(df,symbol,charts):

    p=load('Parabolic SAR backtest.py')
    data=df.drop(columns=['Adj Close','Volume']).reset_index()
    sig=p.signal_generation(data,p.parabolic_sar)
    sig.set_index('Date',inplace=True)
    p.plot(sig.iloc[-250:],symbol)

    return sig['Close'],sig['positions'],'af 0.02 to 0.2, long only'


def run_rsi(df,symbol,charts):

    r=load('RSI Pattern Recognition backtest.py')
    sig=r.signal_generation(df.copy(),r.rsi,n=14)
    r.plot(sig.iloc[-250:],symbol)

    return sig['Close'],sig['positions'],'long below 30, short above 70'


def run_rsi_pattern(df,symbol,charts):

    r=load('RSI Pattern Recognition backtest.py')
    sig=r.pattern_recognition(df.copy(),r.rsi,lag=14)
    exposure=sig['signals'].cumsum().clip(-1,0)
    if (sig['signals']!=0).any():
        try:
            r.pattern_plot(sig,symbol)
        except Exception:
            pass

    return sig['Close'],exposure,'head-shoulder top on RSI, short'


def run_shooting_star(df,symbol,charts):

    s=load('Shooting Star backtest.py')
    data=df.reset_index()
    sig=s.signal_generation(data,s.shooting_star)
    stars=sig.index[sig['signals']==-1]
    last=stars[-1] if len(stars)>0 else sig.index[-1]
    s.plot(sig.loc[max(last-5,0):last+10].reset_index(drop=True),symbol)
    sig.set_index('Date',inplace=True)

    #the pattern needs the next candle to confirm
    #so the short can only be opened one bar later
    exposure=sig['positions'].clip(-1,0).shift(1).fillna(0)

    return sig['Close'],exposure,'short, 5% stop, 7 day max hold, entry delayed 1 day'


def run_bollinger(df,symbol,charts):

    b=load('Bollinger Bands Pattern Recognition backtest.py')
    data=pd.DataFrame({'date':df.index,'price':df['Close'].values})
    sig=b.signal_generation(data,b.bollinger_bands,alpha=0.02,beta=0.015)
    b.plot(sig.copy())
    sig.index=df.index

    return sig['price'],sig['signals'].cumsum().clip(0,1),'double bottom, alpha 2% beta 1.5%, long only'


def run_monte_carlo(df,symbol,charts):

    mc=load('Monte Carlo project/Monte Carlo backtest.py')
    random.seed(0)
    forecast_horizon,d,pick=mc.monte_carlo(df)
    mc.plot(df,forecast_horizon,d,pick,symbol)

    split=len(df)-forecast_horizon-1
    predicted=d[pick][-1]-d[pick][split]
    actual=df['Close'].iloc[-1]-df['Close'].iloc[split]

    return {
        'split':df.index[split].strftime('%Y-%m-%d'),
        'predicted':'up' if predicted>0 else 'down',
        'actual':'up' if actual>0 else 'down',
        'correct':(predicted>0)==(actual>0),
    }


def run_pair(pair,stdate,eddate,charts):

    pr=load('Pair trading backtest.py')
    asset1=yahoo_data.download(pair[0],start=stdate,end=eddate)
    asset2=yahoo_data.download(pair[1],start=stdate,end=eddate)
    common=asset1.index.intersection(asset2.index)
    asset1,asset2=asset1.loc[common],asset2.loc[common]

    sig=pr.signal_generation(asset1,asset2,pr.EG_method)
    valid=sig['z'].dropna()
    if len(valid)>0:
        pr.plot(sig[valid.index[0]:],asset1.attrs['ticker'],asset2.attrs['ticker'])

    return sig,asset1.attrs['ticker'],asset2.attrs['ticker']


def run_dual_thrust(ticker,symbol,charts):

    dt=load('Dual Thrust backtest.py')
    df=yahoo_data.download_intraday(ticker)
    df['in session']=dt.session_mask(df,symbol)
    intraday=dt.min2day(df,'Close',5)
    sig=dt.signal_generation(df,intraday,0.5,'Close',5)
    dt.plot(sig,intraday,'Close')

    return sig['Close'],sig['cumsum'].clip(-1,1),'5 day range, k=0.5, flat at close'


def run_london_breakout(ticker,symbol,charts):

    lb=load('London Breakout backtest.py')
    df=yahoo_data.download_intraday(ticker)
    fx=yahoo_data.is_fx(symbol)
    sig=lb.signal_generation(df,lb.london_breakout,fx=fx)
    lb.plot(sig)

    note='tokyo hour range, london session' if fx else '30 min opening range breakout'
    return sig['Close'],sig['cumsum'].clip(-1,1),note


def markdown_table(df):

    rows=[list(df.columns)]+df.astype(str).values.tolist()
    lines=['| '+' | '.join(r)+' |' for r in rows]
    lines.insert(1,'|'+'|'.join(['---']*len(df.columns))+'|')

    return '\n'.join(lines)


# In[4]: overview charts

def overview(results,folder,symbol):

    #total return per daily strategy, one series so one colour
    #intraday runs only cover 60 days, they stay in the table
    table=pd.DataFrame([r for r in results if r['frequency']=='daily']).sort_values('total return')
    if len(table)==0:
        return
    fig,ax=plt.subplots(figsize=(9,0.45*len(table)+1.5))
    ax.barh(table['strategy'],table['total return']*100,color='#2a78d6',height=0.6)
    for y,value in enumerate(table['total return']):
        ax.annotate('%+.1f%%'%(value*100),(max(value,0)*100,y),xytext=(4,0),
                    textcoords='offset points',va='center',ha='left',
                    fontsize=9,color='#52514e')
    bh=table.loc[table['strategy']!='Pair Trading','buy and hold']
    if len(bh)>0:
        bh=bh.iloc[0]*100
        ax.axvline(bh,color='#0b0b0b',linestyle='--',linewidth=1)
        ax.annotate('buy & hold %+.1f%%'%bh,(bh,len(table)-0.5),xytext=(-4,0),
                    textcoords='offset points',fontsize=9,color='#0b0b0b',ha='right')
    ax.axvline(0,color='#b5b3ad',linewidth=1)
    ax.set_xlabel('total return (%)')
    ax.set_title('%s  daily strategies, total return %s to %s'%(
        symbol,table['start'].min(),table['end'].max()),loc='left')
    for side in ['top','right']:
        ax.spines[side].set_visible(False)
    ax.grid(axis='x',color='#e6e5e0',linewidth=0.8)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(folder,'returns.png'),dpi=120,bbox_inches='tight')
    plt.close(fig)

    #equity of the best three daily strategies against buy and hold
    daily=[r for r in results if r['frequency']=='daily' and r['strategy']!='Pair Trading']
    if len(daily)==0:
        return
    best=sorted(daily,key=lambda r:r['total return'],reverse=True)[:3]
    colors=['#2a78d6','#eb6834','#1baf7a']
    fig,ax=plt.subplots(figsize=(10,5))
    price=daily[0]['_price']
    ax.plot(price/price.iloc[0],color='#52514e',linewidth=2,label='Buy & Hold')
    for r,c in zip(best,colors):
        ax.plot(r['_equity'],color=c,linewidth=2,label=r['strategy'])
    ax.set_ylabel('growth of 1')
    ax.set_title('%s  top 3 daily strategies vs buy & hold'%symbol,loc='left')
    ax.legend(frameon=False,loc='upper left')
    for side in ['top','right']:
        ax.spines[side].set_visible(False)
    ax.grid(color='#e6e5e0',linewidth=0.8)
    fig.savefig(os.path.join(folder,'equity.png'),dpi=120,bbox_inches='tight')
    plt.close(fig)


# In[5]:

def main():

    today=datetime.date.today().strftime('%Y-%m-%d')
    parser=argparse.ArgumentParser(description='run every backtest on one ticker')
    parser.add_argument('ticker',nargs='?',default='2330',
                        help='yahoo ticker, taiwan stocks as 2330 / 6488 / 0050')
    parser.add_argument('start',nargs='?',default='2016-01-01')
    parser.add_argument('end',nargs='?',default=today)
    parser.add_argument('--pair',nargs=2,default=None,
                        help='two tickers for pair trading, default 2881 2882 for taiwan, NVDA AMD otherwise')
    args=parser.parse_args()

    df=yahoo_data.download(args.ticker,start=args.start,end=args.end)
    symbol=df.attrs['ticker']
    folder=os.path.join(ROOT,'output',symbol)
    os.makedirs(folder,exist_ok=True)
    charts=ChartSaver(folder)

    taiwan=yahoo_data.is_taiwan(symbol)
    pair=args.pair or (['2881','2882'] if taiwan else ['NVDA','AMD'])

    print('%s  %s to %s  %d trading days'%(symbol,df.index[0].date(),df.index[-1].date(),len(df)))
    print('trading cost per side: %.4f%% daily, %.4f%% intraday'%(
          cost_per_side(symbol,False)*100,cost_per_side(symbol,True)*100))

    #verified on GBPUSD=X and EURUSD=X: where the close sits in the day's range
    #correlates about -0.7 with the next day's return, impossible for real bars
    fx_daily=yahoo_data.is_fx(symbol)
    if fx_daily:
        print('WARNING',yahoo_data.FX_DAILY_WARNING)

    daily_runs=[
        ('MACD',run_macd),
        ('Awesome Oscillator',run_awesome),
        ('Heikin-Ashi',run_heikin_ashi),
        ('Parabolic SAR',run_parabolic_sar),
        ('RSI',run_rsi),
        ('RSI Head-Shoulder',run_rsi_pattern),
        ('Shooting Star',run_shooting_star),
        ('Bollinger Bands',run_bollinger),
    ]
    intraday_runs=[
        ('Dual Thrust',run_dual_thrust),
        ('London Breakout',run_london_breakout),
    ]

    results=[]
    failures=[]
    number=0

    for name,func in daily_runs:
        number+=1
        charts.start(number,name)
        print('running',name)
        try:
            price,exposure,note=func(df,symbol,charts)
            if fx_daily and name in HIGH_LOW_STRATEGIES:
                note='UNRELIABLE on yahoo fx daily high/low. '+note
            r=evaluate(name,'daily',price,exposure,cost_per_side(symbol,False),note)
            r['_price']=price
            results.append(r)
        except Exception as e:
            failures.append((name,repr(e)))
        plt.close('all')

    number+=1
    charts.start(number,'Pair Trading')
    print('running Pair Trading')
    try:
        sig,t1,t2=run_pair(pair,args.start,args.end,charts)
        r=evaluate('Pair Trading','daily',sig['asset1'],sig['signals1'],
                   cost_per_side(t1,False),'%s / %s, Engle-Granger, 250 day window'%(t1,t2),
                   leg2=(sig['asset2'],sig['signals2']))
        r['buy and hold']=(sig['asset1'].iloc[-1]/sig['asset1'].iloc[0]+
                           sig['asset2'].iloc[-1]/sig['asset2'].iloc[0])/2-1
        results.append(r)
    except Exception as e:
        failures.append(('Pair Trading',repr(e)))
    plt.close('all')

    for name,func in intraday_runs:
        number+=1
        charts.start(number,name)
        print('running',name)
        try:
            price,exposure,note=func(args.ticker,symbol,charts)
            results.append(evaluate(name,'5 min',price,exposure,cost_per_side(symbol,True),note))
        except Exception as e:
            failures.append((name,repr(e)))
        plt.close('all')

    number+=1
    charts.start(number,'Monte Carlo')
    print('running Monte Carlo')
    try:
        forecast=run_monte_carlo(df,symbol,charts)
    except Exception as e:
        forecast=None
        failures.append(('Monte Carlo',repr(e)))
    plt.close('all')

    #summary table
    columns=['strategy','frequency','start','end','total return','annual return',
             'buy and hold','sharpe','max drawdown','trades','win rate','time in market','note']
    summary=pd.DataFrame(results)[columns]
    summary.to_csv(os.path.join(folder,'summary.csv'),index=False,encoding='utf-8-sig')

    pct=['total return','annual return','buy and hold','max drawdown','win rate','time in market']
    shown=summary.copy()
    for c in pct:
        shown[c]=shown[c].map(lambda x:'' if pd.isna(x) else '%.1f%%'%(x*100))
    shown['sharpe']=shown['sharpe'].map(lambda x:'' if pd.isna(x) else '%.2f'%x)

    lines=['# %s backtest summary'%symbol,'']
    if fx_daily:
        lines+=['WARNING: '+yahoo_data.FX_DAILY_WARNING,'']
    lines+=[
           'Period %s to %s, trading cost per side %.4f%% (daily) / %.4f%% (intraday).'%(
               df.index[0].date(),df.index[-1].date(),
               cost_per_side(symbol,False)*100,cost_per_side(symbol,True)*100),'',
           markdown_table(shown),'']
    if forecast is not None:
        lines.append('Monte Carlo: trained until %s, predicted %s, actual %s (%s).'%(
            forecast['split'],forecast['predicted'],forecast['actual'],
            'correct' if forecast['correct'] else 'wrong'))
    for name,error in failures:
        lines.append('FAILED %s: %s'%(name,error))
    with open(os.path.join(folder,'summary.md'),'w',encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')

    overview(results,folder,symbol)

    pd.set_option('display.width',200)
    print()
    print(shown.drop(columns=['note']).to_string(index=False))
    if forecast is not None:
        print('\nMonte Carlo direction: predicted %s, actual %s'%(forecast['predicted'],forecast['actual']))
    for name,error in failures:
        print('FAILED',name,error)
    print('\nresults saved to',folder)


if __name__ == '__main__':
    main()
