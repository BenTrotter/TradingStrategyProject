import random
import operator
import math
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

file = 'MSFT.csv'

def ma(date,window):

    panda = pd.read_csv(file) # Maybe set MSFT.csv variable to a global variable
    mas = panda["SMA "+str(window)+".0"]
    dates = panda["Date"]
    dict1 = dict(zip(dates,mas))
    ma = dict1[date]
    return ma

def rsi(date,window):

    panda = pd.read_csv(file)
    rsi = panda["RSI "+str(window)+".0"]
    dates = panda["Date"]
    dict1 = dict(zip(dates,rsi))
    RSI = dict1[date]
    return RSI

def ema(date,window):

    panda = pd.read_csv(file)
    emas = panda["EMA "+str(window)+".0"]
    dates = panda["Date"]
    dict1 = dict(zip(dates,emas))
    ema = dict1[date]
    return ema

def macd(date,s,f,sig):

    panda = pd.read_csv(file)
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

    panda = pd.read_csv(file)
    SO = panda["SO "+str(window)]
    sig = panda["%D 3-"+str(window)] # Maybe make this a variable that is explored by the algorithm
    dates = panda["Date"]
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
    panda["Date"] = pd.to_datetime(panda['Date']) # date column to datetime
    # retrieve dates from series between start and end date
    mask = (panda['Date'] > startDate) & (panda['Date'] <= endDate)
    panda = panda.loc[mask]
    prices = panda["Close"]
    panda = panda.astype(str) # make everything in the panda to a string
    dates = panda['Date']
    combined = dict(zip(dates, round(prices,2)))

    return combined


def plot_action(data,whichSplits):

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

    locs = ax.get_xticks()
    print(len(locs))
    split = len(locs)/len(whichSplits)


    ax.axvspan(8, 14, alpha=0.5, color='green')
    count = 1
    a = 0
    b = split
    for i in whichSplits:
        if i == True:
            ax.axvspan(a, b, alpha=0.5, color='green')
        else:
            ax.axvspan(a, b, alpha=0.5, color='red')
        count += 1
        a+=split
        b+= split

    # plt.fill_between(x, y1, y2,
    #              facecolor="orange", # The fill color
    #              color='blue',       # The outline color
    #              alpha=0.2)
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

    startDate = '2015-01-01' # Start date of the overall trading window.
    endDate = '2021-01-01' # End date of the overall trading window.
    shares = 0 # The number of shares that the trader currently owns.
    position = False # Whether the trader is currently invested in the stock.
    startingBalance = 1000 # The starting amount that the trader will trade with.
    balance = startingBalance # The current balance or equity of the trader.
    numTrades = 0 # The number of trades that have been executed.
    iter = 0
    findStart = False # Bool to help find the first tradable start date.
    # Converting date strings to  datetime objects
    startDay = datetime.strptime(startDate,'%Y-%m-%d')
    endDay = datetime.strptime(endDate,'%Y-%m-%d')
    pcSplit = 72 # The number of intervals to split the trading window into for PC
    # Get the interval in days for PC
    interval = splitTrainingPeriod(startDay, endDay, pcSplit)
    # Get the dates that the PC count needs to be checked and updated.
    print(startDay)
    print(endDay)
    performanceConsistencyDates = getPCUpdateDates(startDay, endDay, interval)
    # Gets a dict of date and price as keys and values.
    priceData = getPriceDataDict(file,startDay,endDay)
    whichSplits = []

    riskFreeRate = 0.05
    rateOfReturn = []
    dailyReturn = []


    buys = []
    sells = []
    data = {}

    # todo: need to test that riskExposure is producing accurate results
    riskExposure = 0 # The num days that a trader is in the market.
    pcCount = 0 # The count for the performancy consistency value.
    pcIter = 0 # The index for the performanceConsistencyDatess
    answer = 1 # Initialising answer to prevent zero division error.

    print('Interval: ', interval)
    numTDays = len(priceData)

    for date, price in priceData.items(): # Need to make within the sim time frame
        # to start the sim from the start date
        if datetime.strptime(date,'%Y-%m-%d') < startDay and not findStart:
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
        if datetime.strptime(date, '%Y-%m-%d') >= performanceConsistencyDates[pcIter]:
            print('PC check: ', pcIter,' on ',date)
            percentIncPriceTest = ((price-oldPrice)/oldPrice)*100
            percentIncBalanceStrategy =  ((balance-oldBalance)/oldBalance)*100
            if percentIncPriceTest < percentIncBalanceStrategy:
                pcCount += 1
                whichSplits.append(True)
            else:
                whichSplits.append(False)
            # print("\nPC Update: ",date)
            # print("Increase in price: ",round(percentIncPriceTest,2))
            # print("Increase in balance: ",round(percentIncBalanceStrategy,2),"\n")
            oldPrice = price
            oldBalance = balance
            pcIter+=1

        #action = operator.or_(operator.lt(ema(date, 12), ema(date, 26)), operator.or_(operator.gt(0, so(date, 7)), operator.gt(ma(date, 200), ma(date, 10))))
        action = operator.or_(operator.not_(operator.gt(ma(date,5),ma(date,10))), operator.gt(0,macd(date,26,12,9)))

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
            balanceSave = balance
            buys.append(date)
            data[date] = [price,True,False]
            print("----- BUY £",round(balance,2)," -----")
            print('On date: ',date)
            print("Bought ",round(shares,4),' shares at price ',price,'\n')

        elif sell and shares > 0:
            position = False
            balance = round(shares*price, 2)
            percentReturnOfTrade = round((((balance - balanceSave)/balanceSave)*100),2)
            rateOfReturn.append(percentReturnOfTrade)
            profit = balance - oldAmount
            sells.append(date)
            data[date] = [price,False,True]
            print("----- SELL £",round(balance,2)," -----")
            print("Profit from indivdual sale: ",round(profit,2)," or ",percentReturnOfTrade,'%')
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
        if datetime.strptime(date,'%Y-%m-%d') >= endDay:
            if position:
                balance = round(shares*price, 2)
                percentReturnOfTrade = round((((balance - balanceSave)/balanceSave)*100),2)
                rateOfReturn.append(percentReturnOfTrade)
            bhBalance = bhShares*priceData[oldDate]
            break

        oldDate = date

        if position:
            try:
                dailyReturn.append((((price-oldP)/oldP)*100)-0.05/numTDays)
            except UnboundLocalError:
                dailyReturn.append(0)
        else:
            dailyReturn.append(0-0.05/numTDays) # add risk free rate of 0.05 as a global variable

        oldP = price


    # Final peformance consitency check on final iterval
    if(pcIter != pcSplit):
        percentIncPriceTest = ((price-oldPrice)/oldPrice)*100.0
        percentIncBalanceStrategy =  ((balance-oldBalance)/oldBalance)*100
        if percentIncPriceTest < percentIncBalanceStrategy:
            pcCount += 1
            whichSplits.append(True)
        else:
            whichSplits.append(False)
        pcIter+=1

    # print("\nPC Update: ",date)
    # print("Increase in price: ",round(percentIncPriceTest,2))
    # print("Increase in equity: ",round(percentIncBalanceStrategy,2),"\n")
    # pcIter+=1

    # Check to ensure every interval as been included in the PC count
    if(pcIter != pcSplit):
        print("\nERROR: Not all pc intervals have been calculated.\n")


    aveReturn = sum(rateOfReturn) / len(rateOfReturn)
    stdRateOfReturn = numpy.std(rateOfReturn)
    sharpe = round((math.sqrt(len(rateOfReturn)) * ((aveReturn - riskFreeRate) / stdRateOfReturn)),2)
    sharpe1 = round(((aveReturn - riskFreeRate)/stdRateOfReturn),2)
    sharpe2 = round(((aveReturn - ((riskFreeRate/252)*riskExposure))/stdRateOfReturn),2)

    aveDailyReturn = sum(dailyReturn)/len(dailyReturn)
    stdDailyRateOfReturn = numpy.std(dailyReturn)
    sharpe3 = round(numpy.sqrt(numTDays) * aveDailyReturn / stdDailyRateOfReturn,2)

    answer = ((balance - startingBalance)/startingBalance)*100
    print("Percentage increase by ", date," is ", round(answer,2))
    print("Number of trades: ",numTrades)
    print("Risk exposure: ",riskExposure)
    print("PC count: ",pcCount)
    print("Sharpe Ratio: ",sharpe1,' or ', sharpe,' or ',sharpe2,' or ',sharpe3)

    if numTrades == 0:
        numTrades = 100
        riskExposure = 100

    for d in whichSplits:
        print("Split: ",d)

    return data, whichSplits


if __name__ == "__main__":
    prDa, splits = simulation()
    plot_action(prDa,splits)
