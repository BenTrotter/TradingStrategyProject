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

file = 'TSLA.csv'

def ma(date,window):

    panda = pd.read_csv(file) # Maybe set MSFT.csv variable to a global variable
    mas = panda["SMA "+str(window)+".0"]
    dates = panda["Datetime"]
    dict1 = dict(zip(dates,mas))
    ma = dict1[date]
    return ma

def rsi(date,window):

    panda = pd.read_csv(file)
    rsi = panda["RSI "+str(window)+".0"]
    dates = panda["Datetime"]
    dict1 = dict(zip(dates,rsi))
    RSI = dict1[date]
    return RSI

def ema(date,window):

    panda = pd.read_csv(file)
    emas = panda["EMA "+str(window)+".0"]
    dates = panda["Datetime"]
    dict1 = dict(zip(dates,emas))
    ema = dict1[date]
    return ema

def macd(date,s,f,sig):

    panda = pd.read_csv(file)
    macd = panda["MACD "+str(s)+str(f)]
    sig = panda["MACD Signal "+str(sig)+str(s)+str(f)]
    dates = panda["Datetime"]
    dict1 = dict(zip(dates,macd))
    dict2 = dict(zip(dates,sig))
    macd1 = dict1[date]
    sig1 = dict2[date]
    ans = macd1 - sig1
    return ans

def so(date,window):

    panda = pd.read_csv(file)
    SO = panda["SO "+str(window)]
    sig = panda["%D 3-"+str(window)] # Maybe make this a variable that is explored by the algorithm
    dates = panda["Datetime"]
    dict1 = dict(zip(dates,SO))
    dict2 = dict(zip(dates,sig))
    so = dict1[date]
    sig1 = dict2[date]
    ans = so - sig1 # Check this is the right way round. It may not matter.
    return ans

def if_then_else(input, output1, output2):
    return output1 if input else output2


def splitTrainingPeriod(startD, endD, k):
    """
    Splits the overall trading window by an integer k. The function then returns
    the interval of days that the window has been split into.
    """
    days = endD - startD
    interval = days/k
    return interval

def getPCUpdateDates(startDay,endDay,interval):
    """
    Returns a list containing all of the dates that the PC count must make a check
    and update its value.
    """
    performanceConsistencyDates = []
    checkDay = startDay

    while(checkDay < endDay):
        checkDay += interval
        performanceConsistencyDates.append(checkDay)

    return performanceConsistencyDates

def getPriceDataDict(csv,startDate,endDate):
    """
    Gets a dict of date and price as keys and values. It trims the dict
    to only contain the relevant data between certain dates.
    """
    panda = pd.read_csv(csv)
    panda["Datetime"] = pd.to_datetime(panda['Datetime']) # date column to datetime
    # retrieve dates from series between start and end date
    mask = (panda['Datetime'] > startDate) & (panda['Datetime'] <= endDate)
    panda = panda.loc[mask]
    prices = panda["Close"]
    panda = panda.astype(str) # make everything in the panda to a string
    dates = panda['Datetime']
    combined = dict(zip(dates, round(prices,2)))

    return combined


def plot_action(data):

    buyPlt = []
    sellPlt = []
    pricesPlt=[]
    dates = []

    for k,v in data.items():
        dates.append(k)
        pricesPlt.append(v)

        if v[1] == True:
            buyPlt.append(k)
        else:
            buyPlt.append(0)

        if v[2] == True:
            sellPlt.append(k)
        else:
            sellPlt.append(0)

    datesNp = numpy.array(dates)
    pricesNp = numpy.array(pricesPlt)
    buyNp = numpy.array(buyPlt)
    sellNp = numpy.array(sellPlt)

    fig, ax = plt.subplots()

    plt.plot_date(dates, pricesNp, 'k', label='Closing Price')
    plt.plot(buyNp, pricesNp,'^', markersize=5, color='g')
    plt.plot(sellNp, pricesNp,'v', markersize=5, color='r')

    every_nth = 12
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    plt.gcf().autofmt_xdate()

    plt.title('Microsoft Price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')

    plt.tight_layout()

    plt.show()

def simulation():

    startDate = "2021-03-19 08:00:00" # Start date of the overall trading window.
    endDate = "2021-03-19 17:00:00" # End date of the overall trading window.
    shares = 0 # The number of shares that the trader currently owns.
    position = False # Whether the trader is currently invested in the stock.
    startingBalance = 1000 # The starting amount that the trader will trade with.
    balance = startingBalance # The current balance or equity of the trader.
    numTrades = 0 # The number of trades that have been executed.
    iter = 0
    findStart = False # Bool to help find the first tradable start date.
    # Converting date strings to  datetime objects
    startDay = datetime.strptime(startDate,'%Y-%m-%d %H:%M:%S')
    endDay = datetime.strptime(endDate,'%Y-%m-%d %H:%M:%S')
    pcSplit = 4 # The number of intervals to split the trading window into for PC
    # Get the interval in days for PC
    interval = splitTrainingPeriod(startDay, endDay, pcSplit)
    # Get the dates that the PC count needs to be checked and updated.
    print(startDay)
    print(endDay)
    performanceConsistencyDates = getPCUpdateDates(startDay, endDay, interval)
    # Gets a dict of date and price as keys and values.
    priceData = getPriceDataDict(file,startDay,endDay)

    buys = []
    sells = []
    data = {}

    # todo: need to test that riskExposure is producing accurate results
    riskExposure = 0 # The num days that a trader is in the market.
    pcCount = 0 # The count for the performancy consistency value.
    pcIter = 0 # The index for the performanceConsistencyDatess
    answer = 1 # Initialising answer to prevent zero division error.

    print('Interval: ', interval)

    for date, price in priceData.items(): # Need to make within the sim time frame
        # to start the sim from the start date
        if datetime.strptime(date,'%Y-%m-%d %H:%M:%S') < startDay and not findStart:
            continue
        # calculating the b&h strategy at start date and the first PC interval.
        elif not findStart:
            startD = date
            findStart = True
            bhShares = startingBalance / priceData[date]
            oldPrice = price
            oldBalance = startingBalance

        if iter == 0:
            oldDate = date
            iter += 1
            continue

        # PC update if the correct date has been reached
        if datetime.strptime(date, '%Y-%m-%d %H:%M:%S') >= performanceConsistencyDates[pcIter]:
            print('PC check: ', pcIter,' on ',date)
            percentIncPriceTest = ((price-oldPrice)/oldPrice)*100
            percentIncBalanceStrategy =  ((balance-oldBalance)/oldBalance)*100
            if percentIncPriceTest < percentIncBalanceStrategy:
                pcCount += 1
            # print("\nPC Update: ",date)
            # print("Increase in price: ",round(percentIncPriceTest,2))
            # print("Increase in balance: ",round(percentIncBalanceStrategy,2),"\n")
            oldPrice = price
            oldBalance = balance
            pcIter+=1

        action = operator.not_(operator.not_(operator.gt(0, macd(date, 26, 12, 9))))
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
            numTrades += 1
            position = True
            shares = balance/price
            oldAmount = balance
            balance = round(price*shares,2)
            buys.append(date)
            data[date] = [price,True,False]
            print("----- BUY £",round(balance,2)," -----")
            print('On date: ',date)
            print("Bought ",round(shares,4),' shares at price ',price,'\n')

        elif sell and shares > 0:
            position = False
            balance = round(shares*price, 2)
            profit = balance - oldAmount
            sells.append(date)
            data[date] = [price,False,True]
            print("----- SELL £",round(balance,2)," -----")
            print("Profit from indivdual sale: ",round(profit,2))
            print('On date: ',date)
            print("Sold ",round(shares,4),' shares at price ',price,'\n')
            shares = 0

        elif shares != 0:
            balance = round(price*shares,2)
            data[date] = [price,False,False]
        else:
            data[date] = [price,False,False]

        if position == True:
            riskExposure += 1

        # Ends the simulation at the end date
        if datetime.strptime(date,'%Y-%m-%d %H:%M:%S') >= endDay:
            bhBalance = bhShares*priceData[oldDate]
            break

        oldDate = date

    # Final peformance consitency check on final iterval
    if(pcIter != pcSplit):
        percentIncPriceTest = ((price-oldPrice)/oldPrice)*100
        percentIncBalanceStrategy =  ((balance-oldBalance)/oldBalance)*100
        if percentIncPriceTest < percentIncBalanceStrategy:
            pcCount += 1
        pcIter+=1

    # print("\nPC Update: ",date)
    # print("Increase in price: ",round(percentIncPriceTest,2))
    # print("Increase in equity: ",round(percentIncBalanceStrategy,2),"\n")
    # pcIter+=1

    # Check to ensure every interval as been included in the PC count
    if(pcIter != pcSplit):
        print("\nERROR: Not all pc intervals have been calculated.\n")


    answer = ((balance - startingBalance)/startingBalance)*100
    print("Percentage increase by ", date," is ", round(answer,2))
    print("Number of trades: ",numTrades)
    print("Risk exposure: ",riskExposure)
    print("PC count: ",pcCount)

    if numTrades == 0:
        numTrades = 100
        riskExposure = 100

    return data


if __name__ == "__main__":
    prDa = simulation()
    plot_action(prDa)
