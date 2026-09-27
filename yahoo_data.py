# -*- coding: utf-8 -*-
"""
Shared Yahoo Finance loader for every backtest in this repository.

- One place to call yfinance, always returning flat
  Open/High/Low/Close/Adj Close/Volume columns
- Taiwan stocks can be given as bare codes:
  '2330' -> '2330.TW' (TWSE), '6488' -> '6488.TWO' (TPEx),
  '0050' / '00878' ETFs work the same way, '^TWII' is the TAIEX index
- Command line helper so every script accepts
  python "<script>.py" [ticker] [start] [end]
"""

import logging
import re
import sys

import pandas as pd
import yfinance as yf


TW_CODE=re.compile(r'^\d{4,6}[A-Z]?$')


def _fetch(ticker,**kwargs):

    #yfinance logs an error for every symbol that does not exist
    #probing .TW before .TWO would spam the console, so silence it here
    logger=logging.getLogger('yfinance')
    level=logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        df=yf.download(ticker,auto_adjust=False,multi_level_index=False,
                       progress=False,**kwargs)
    finally:
        logger.setLevel(level)

    return df


def is_taiwan(ticker):

    t=ticker.upper()
    return bool(TW_CODE.match(t)) or t.endswith('.TW') \
        or t.endswith('.TWO') or t=='^TWII' or t=='^TWOII'


def download(ticker,start=None,end=None,interval='1d',period=None):

    t=ticker.strip().upper()

    #bare taiwan code, try listed market first then otc market
    if TW_CODE.match(t):
        candidates=[t+'.TW',t+'.TWO']
    else:
        candidates=[t]

    kwargs={'interval':interval}
    if period is not None:
        kwargs['period']=period
    else:
        kwargs['start']=start
        kwargs['end']=end

    for symbol in candidates:
        df=_fetch(symbol,**kwargs)
        df=df.dropna(subset=['Close'])
        if len(df)>0:
            df.attrs['ticker']=symbol
            return df

    raise ValueError(f'Yahoo Finance has no data for {ticker} '
                     f'({", ".join(candidates)}) in the requested period')


def download_intraday(ticker,interval='5m',period='60d'):
    """
    Intraday bars in the exchange's local clock time, timezone dropped
    yahoo keeps 5 minute bars for the last 60 days only
    """

    df=download(ticker,interval=interval,period=period)
    symbol=df.attrs['ticker']
    df.index=df.index.tz_localize(None)
    df.attrs['ticker']=symbol

    return df


def is_fx(ticker):

    return ticker.upper().endswith('=X')


def resolve(ticker):
    """Return the Yahoo symbol a user input maps to, e.g. '2330' -> '2330.TW'."""

    return download(ticker,period='5d').attrs['ticker']


def cli_args(ticker,stdate,eddate):
    """
    Let every script be run as
    python "<script>.py" [ticker] [start] [end]
    anything omitted keeps the script's original default
    """

    args=sys.argv[1:]
    if len(args)>0:
        ticker=args[0]
    if len(args)>1:
        stdate=args[1]
    if len(args)>2:
        eddate=args[2]

    return ticker,stdate,eddate


def trading_cost(ticker):
    """
    Cost charged per unit of position change, used by run_all.py
    taiwan: 0.1425% broker fee each side plus 0.3% securities tax on sells,
    split evenly across both sides -> 0.2925% per side
    elsewhere the original scripts assume frictionless trading
    """

    return 0.002925 if is_taiwan(ticker) else 0.0
