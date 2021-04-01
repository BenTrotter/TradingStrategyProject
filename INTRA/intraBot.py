import numpy as np
import csv
import math
import pandas as pd
import yfinance as yf
import sys
import os
from datetime import datetime, timedelta
import time
from intraPreload import *
import operator

import deap
from importlib import reload
reload(deap)

from deap import algorithms, base, creator, tools, gp
from deap.benchmarks.tools import diversity, convergence, hypervolume

import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import alpaca_trade_api as tradeapi
# from alpaca_trade_api.rest import TimeFrame

alpaca_endpoint = 'https://paper-api.alpaca.markets'
api = tradeapi.REST('PKLWLP03YXQEN9SKK7Y1','rPrIeju8H8w9JdsZKGpaz0sCmFPq5EXv1Z9JpV3V',alpaca_endpoint)

# Global
file1 = "TSLA.csv"
ticker = "TSLA"
global alpacaTicker
alpacaTicker = "TSLA"
global position
position = False
global password
password = "Tinker4242" # hide if going live
global trade
trade = False
global numShares
numShares = 1

class pd_float(object):
    pass

class pd_bool(object):
    pass
class position_bool(object):
    pass

class pd_int(object):
    pass

class ema_int(object):
    pass

class ma_int(object):
    pass

class rsi_int(object):
    pass
class rsi_bounds(object):
    pass
class rsi_ans(object):
    pass
class rsi_lt(object):
    pass
class rsi_gt(object):
    pass

class macd_sig(object):
    pass
class macd_f(object):
    pass
class macd_s(object):
    pass
class macd_ans(object):
    pass
class macd_bounds(object):
    pass

class so_int(object):
    pass
class so_ans(object):
    pass
class so_bounds(object):
    pass

def ma(date,window):

    panda = pd.read_csv(file1) # Maybe set MSFT.csv variable to a global variable
    mas = panda["SMA "+str(window)+".0"]
    dates = panda["Unnamed: 0"]
    dict1 = dict(zip(dates,mas))
    ma = dict1[date]
    return ma

def rsi(date,window):

    panda = pd.read_csv(file1)
    rsi = panda["RSI "+str(window)+".0"]
    dates = panda["Unnamed: 0"]
    dict1 = dict(zip(dates,rsi))
    RSI = dict1[date]
    return RSI

def ema(date,window):

    panda = pd.read_csv(file1)
    emas = panda["EMA "+str(window)+".0"]
    dates = panda["Unnamed: 0"]
    dict1 = dict(zip(dates,emas))
    ema = dict1[date]
    return ema

def macd(date,s,f,sig):

    panda = pd.read_csv(file1)
    macd = panda["MACD "+str(s)+str(f)]
    sig = panda["MACD Signal "+str(sig)+str(s)+str(f)]
    dates = panda["Unnamed: 0"]
    dict1 = dict(zip(dates,macd))
    dict2 = dict(zip(dates,sig))
    macd1 = dict1[date]
    sig1 = dict2[date]
    ans = macd1 - sig1
    return ans

def so(date,window):

    panda = pd.read_csv(file1)
    SO = panda["SO "+str(window)]
    sig = panda["%D 3-"+str(window)] # Maybe make this a variable that is explored by the algorithm
    dates = panda["Unnamed: 0"]
    dict1 = dict(zip(dates,SO))
    dict2 = dict(zip(dates,sig))
    so = dict1[date]
    sig1 = dict2[date]
    ans = so - sig1 # Check this is the right way round. It may not matter.
    return ans

def if_then_else(input, output1, output2):
    return output1 if input else output2

# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", [str,position_bool], pd_bool)
# boolean operators
pset.addPrimitive(if_then_else, [position_bool, rsi_gt, rsi_lt], pd_bool)
pset.addPrimitive(operator.and_, [pd_bool, pd_bool], pd_bool)
pset.addPrimitive(operator.or_, [pd_bool, pd_bool], pd_bool)
pset.addPrimitive(operator.not_, [pd_bool], pd_bool)
pset.addPrimitive(operator.lt, [pd_float, pd_float], pd_bool)
pset.addPrimitive(operator.gt, [pd_float, pd_float], pd_bool)
pset.addPrimitive(operator.lt, [rsi_ans,rsi_bounds], rsi_lt)
pset.addPrimitive(operator.gt, [rsi_ans, rsi_bounds], rsi_gt)
pset.addPrimitive(operator.lt, [macd_bounds, macd_ans], pd_bool) # for above or below zero, maybe make macd_bounds and so_bounds the same thing
pset.addPrimitive(operator.gt, [macd_bounds, macd_ans], pd_bool)
pset.addPrimitive(operator.lt, [so_bounds, so_ans], pd_bool)
pset.addPrimitive(operator.gt, [so_bounds, so_ans], pd_bool)
pset.addPrimitive(ma, [str, ma_int], pd_float)
pset.addPrimitive(ema, [str, ema_int], pd_float)
pset.addPrimitive(rsi, [str, rsi_int], rsi_ans)
pset.addPrimitive(macd, [str, macd_s, macd_f, macd_sig], macd_ans)
pset.addPrimitive(so, [str, so_int], so_ans)

def calvIndicators():

    csv = file1
    removeTrailingTime(ticker,file1)

    mydict = csvDict(csv)
    mydicthl = csvDictHL(csv)

    maWindows = [5, 10, 20, 30, 50, 100, 200]
    for i in maWindows:
        dict = moveAveList(i,mydict)
        addColum('SMA '+str(i)+'.0',dict,csv)

    emaWindows = [5, 12, 26, 30, 50, 100, 200]
    for i in emaWindows:
        dict1 = getEMASeries(mydict,i)
        addColum('EMA '+str(i)+'.0',dict1,csv)

    rsiWindows = [7, 14, 28]
    for i in rsiWindows:
        rsiDict = getRSI(i,mydict)
        addColum('RSI '+str(i)+'.0',rsiDict,csv)

    macdS = [26, 35]
    macdF = [5, 12]
    macdSig = [5, 9]
    for s in macdS:
        for f in macdF:
            macdDictDT,macdDict = getMACDSeries(mydict,s,f)
            addColum('MACD '+str(s)+str(f),macdDict,csv)
            for sig in macdSig:
                macdSignalDict = getMacdSignal(macdDictDT,sig)
                addColum('MACD Signal '+str(sig)+str(s)+str(f),macdSignalDict,csv)

    soWindows = [7,14,28]
    for w in soWindows:
        soDict, soStrDict = getSOSeries(mydicthl,w)
        soDDict = moveAveList(3,soStrDict)
        addColum('SO '+str(w),soDict,csv)
        addColum('%D 3-'+str(w),soDDict,csv)

    trimCSValpaca(csv)
    os.remove(ticker+'.csv')
    os.rename('Trim.csv',ticker+'.csv')


def round_minutes(dt, direction, res):
    new_minute = (dt.minute // res + (1 if direction == 'up' else 0)) * res
    new = dt + timedelta(minutes=new_minute - dt.minute)
    a = new.strftime('%Y-%m-%d %H:%M:00')
    ans = datetime.strptime(a, '%Y-%m-%d %H:%M:00')
    return ans

def job():

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")

    global position

    # Generate a panda from alpaca data
    panda = api.get_barset(alpacaTicker,'5Min',limit=600).df 
     # Turn panda to csv
    panda.to_csv(os.path.abspath(os.getcwd())+'/'+ticker+'.csv')
    # This function to calculate the indicator values on the csv and update
    calvIndicators()
    # Read the updated csv to a panda
    nowData = pd.read_csv(file1)
    # Get the last row date of the panda.
    Date = nowData["Unnamed: 0"][nowData.index[-1]]

    toolbox = base.Toolbox()
    toolbox.register("compile", gp.compile, pset=pset)

    i = operator.and_(if_then_else(position, operator.gt(rsi(Date, 7), 42), operator.lt(rsi(Date, 7), 53)), operator.lt(ma(Date, 50), ma(Date, 10)))
    rule = toolbox.compile(expr=i)

    action = rule(Date,position)

    if action and position == False:
        buy = True
        sell = False
    elif not action and position == False:
        sell = False
        buy = False
    elif action and position == True:
        sell = False
        buy = False
    elif not action and position == True:
        sell = True
        buy = False

    if buy:
        print(current_time,": Buy")
        position = True

        if trade:
            api.submit_order(symbol=alpacaTicker,
            qty=numShares,
            side='buy',
            time_in_force='gtc',
            type='market'
            )

    elif sell:
        print(current_time,": Sell")
        position = False

        if trade:
            api.submit_order(symbol=alpacaTicker,
            qty=numShares,
            side='sell',
            time_in_force='gtc',
            type='market'
            )

    elif position == False:
        print(current_time,": Wait")
        
    else:
        print(current_time,": Hold")

    return

print("Before starting the bot you need to check if the account currently holds a position.")
posit = input("Type y if holds a position and n if not: ")

if posit == 'y' or posit == 'Y':
    position = True
else:
    position = False

while True:
    job()
    time.sleep(300) # wait 5 minutes
