import random
import operator
import csv
import itertools

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import numpy

import deap
from importlib import reload
reload(deap)

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from datetime import datetime, timedelta
import pandas as pd

def ma(date,window):
    startDay = datetime.strptime(date,'%Y-%m-%d')
    panda = pd.read_csv('MSFT.csv') # Maybe set MSFT.csv variable to a global variable
    mas = panda["SMA "+str(window)+".0"]
    dates = panda["Date"]
    dict1 = dict(zip(dates,mas))
    ma = dict1[date]
    return ma

def rsi(date,window):
    startDay = datetime.strptime(date,'%Y-%m-%d')
    panda = pd.read_csv('MSFT.csv')
    rsi = panda["RSI "+str(window)+".0"]
    dates = panda["Date"]
    dict1 = dict(zip(dates,rsi))
    RSI = dict1[date]
    return RSI

def ema(date,window):
    startDay = datetime.strptime(date,'%Y-%m-%d')
    panda = pd.read_csv('MSFT.csv')
    emas = panda["EMA "+str(window)+".0"]
    dates = panda["Date"]
    dict1 = dict(zip(dates,emas))
    ema = dict1[date]
    return ema

def macd(date,s,f,sig):
    startDay = datetime.strptime(date,'%Y-%m-%d')
    panda = pd.read_csv('MSFT.csv')
    macd = panda["MACD "+str(s)+str(f)]
    sig = panda["MACD Signal "+str(sig)+str(s)+str(f)]
    dates = panda["Date"]
    dict1 = dict(zip(dates,macd))
    dict2 = dict(zip(dates,sig))
    macd1 = dict1[date]
    sig1 = dict2[date]
    ans = macd1 - sig1
    return ans

def so(date,window):
    startDay = datetime.strptime(date,'%Y-%m-%d')
    panda = pd.read_csv('MSFT.csv')
    SO = panda["SO "+str(window)]
    sig = panda["%D 3-"+str(window)] # Maybe make this a variable that is explored by the algorithm
    dates = panda["Date"]
    dict1 = dict(zip(dates,SO))
    dict2 = dict(zip(dates,sig))
    so = dict1[date]
    sig1 = dict2[date]
    ans = so - sig1 # Check this is the right way round. It may not matter.
    return ans

def simulation():

    # rule = toolbox.compile(expr=individual)

    startDate = '2020-01-01'
    endDate = '2021-02-02'
    amount = 1000
    shares = 0
    balance = 1000
    returns = 0
    equity = 0
    startDay = datetime.strptime(startDate,'%Y-%m-%d')
    endDay = datetime.strptime(endDate,'%Y-%m-%d')

    panda = pd.read_csv('MSFT.csv')
    dates = panda["Date"]
    prices = panda["Close"]
    combined = dict(zip(dates, round(prices,2)))
    iter = 0
    startCount = 0
    bhBalance = 0
    bhShares = 0
    position = False

    for date, price in combined.items(): # Need to make within the sim time frame
        # to start the sim from the start date

        if datetime.strptime(date,'%Y-%m-%d') < startDay and startCount == 0:
            continue
        # calculating the b&h strategy at start date
        elif startCount == 0:
            startD = date
            startCount = 1
            bhShares = amount / combined[date]


        if iter == 0:
            oldDate = date
            iter += 1
            continue


        action = operator.lt(ema(date, 200), ma(date, 200))

        if action:
            buy = True
            sell = False
        else:
            sell = True
            buy = False

        if buy and position == False:
            position = True
            shares = amount/price
            balance -= amount
            print("----- BUY £",amount," -----")
            print('On date: ',date)
            print("Bought ",round(shares,4),' shares at price ',price,'\n')

        elif sell and shares > 0 and position == True:
            position = False
            balance = shares*price
            profit = balance - amount
            returns += profit
            print("----- SELL £",round(balance,2)," -----")
            print("Profit from indivdual sale: ",round(profit,2))
            print('On date: ',date)
            print("Sold ",round(shares,4),' shares at price ',price,'\n')

        elif shares != 0:
            equity = price*shares

        # to end the sim at the end date

        if datetime.strptime(date,'%Y-%m-%d') >= endDay:
            bhBalance = bhShares*combined[oldDate]
            break

        oldDate = date
        answer = (((returns+equity)-amount)/amount)*100

    return round(answer,2),#, returns+equity

if __name__ == "__main__":
    simulation()
