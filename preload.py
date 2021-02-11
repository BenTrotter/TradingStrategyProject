import numpy as np
import csv
import math
import pandas as pd
import yfinance as yf
import sys
import os
from datetime import datetime, timedelta
# Require python 3.6 or greater as dictionary ordering is used.

def yfDownload(ticker,start,end,gap):
    """
    Downloads historic data from yfinance api and saves the information into
    a csv file in the current directory.

    @param ticker the ticker symbol to download from yfinacne
    @param start the start date for the data
    @param end the end date for the data.
    """
    print("\nDownloading ",ticker," data from Yahoo Finance")
    panda = yf.download(ticker,start,end,interval=gap)
    print("\n")
    panda.to_csv(os.path.abspath(os.getcwd())+'/'+ticker+'.csv')


def csvDict(csvName):
    """
    Opens and reads a csv file into a dictionary. The dates are the keys and
    the closing price is the values.

    @param a string for the name of the csv
    @return a dictionary containing dates as keys and prices as values
    """
    mydict = {}
    with open(csvName, mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            if row[0] == "Date":
                continue
            key = datetime.strptime(row[0], '%Y-%m-%d')
            mydict[key] = round(float(row[4]),2)
    return mydict

def csvDictHL(csvName):
    """
    Opens and reads a csv file into a dictionary. The dates are the keys and
    the high price and low price are the values.

    @param a string for the name of the csv
    @return a dictionary containing dates as keys and prices as values
    """
    mydict = {}
    with open(csvName, mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            if row[0] == "Date":
                continue
            key = datetime.strptime(row[0], '%Y-%m-%d')
            mydict[key] = [round(float(row[2]),2),round(float(row[3]),2),round(float(row[4]),2)]
    return mydict


def findMovingAverage(date,window,data):
    """
    Returns the moving average of from a date. The moving average time period only
    takes into account trading days so weekends are excluded.

    @param date the date from which to find the moving average.
    @param window the moving average time period.
    @param data the dictionary containing the financial data.
    @return an value for the moving average
    """
    day = date
    count = 0
    try:
        while count < window: # Going back finding the start date excluding weekends
            try:
                data[day]
                count+=1
            except KeyError:
                pass
            day -= timedelta(days=1)
        maList = []
        count1 = 0
        day += timedelta(days=1)
        while count1 < count:
            try:
                maList.append(data[day])
                count1 += 1
            except KeyError:
                pass
            day += timedelta(days=1)

        movingAve = round((sum(maList)/len(maList)),2)

    except OverflowError:
        raise OverflowError
        print("\nNot enough previous data to calculate the desired moving average.")
        print("Either change the simulation period or increase the period of the data")
        print("Program terminated\n")
        sys.exit(1)
        raise

    return movingAve


def addColum(colName,dict,csv):
    """
    Adds a new column to the csv file. Must take in a dictionary that has key
    values as the dates in STRING format, not as a datetime object. It uses
    this string to map the value onto the required row. It will then rewrite
    the csv.

    @param colName the string for the name of the new column.
    @param dict the dictionary containing the data to be added as a column.
    @param csv the name of the csv file to add the column.
    """
    df = pd.read_csv(csv)
    df[colName] = df['Date'].map(dict)
    df.to_csv(os.path.abspath(os.getcwd())+'/'+csv,index=False)


def removeRow(csv1,every):

    with open(csv1, mode='r') as inp, open('Remove.csv', mode='w') as out:
        writer = csv.writer(out)
        count = 0
        for row in csv.reader(inp):
            if count == 0:
                writer.writerow(row)
                count = 1
                continue

            if count == every and row[8] != '0.0': #uncomment after the and
                writer.writerow(row)
                count = 1
                continue
            count +=1
            if count > every:
                count = 1


def remove0Cols(csv1):
    with open(csv1, mode='r') as inp, open('Remove0.csv', mode='w') as out:
        writer = csv.writer(out)
        for row in csv.reader(inp):
            zero = False
            for col in row:
                if col == '0.0':
                    zero = True
            if zero == False:
                writer.writerow(row)

def moveAveList(window,data):
    """
    Returns the moving average dictionary by taking in a price data dictionary.
    It uses the data that is available from the dictionary inputted. The early
    The first values will return 0 due to the fact that it isn't possible to
    generate the moving average without the required historic data.

    @param window the moving average time period.
    @param data the dictionary containing the financial data.
    @return an value for the moving average
    """
    count = 0
    maList = {}
    for k,v in data.items():
        count += 1
        if count < window:
            maList[datetime.strftime(k,'%Y-%m-%d')] = 0
            continue
        ans = findMovingAverage(k,window,data)
        maList[datetime.strftime(k,'%Y-%m-%d')] = ans
    return maList


# def getROC(date,window,data):
#     """
#     Returns the rate of change(ROC) for a certain  date. The ROC window only
#     takes into account trading days so weekends are excluded.
#
#     @param date the date to find the ROC for.
#     @param window the ROC time period.
#     @param data the dictionary containing the financial data.
#     @return an value for the ROC as a percentage.
#     """
#     priceNew = data[date]
#     day = date
#     count = 0
#     try:
#         while count <= window: # Going back finding the start date excluding weekends
#             try:
#                 data[day]
#                 count+=1
#             except KeyError:
#                 pass
#             day -= timedelta(days=1)
#
#         priceOld = data[day+timedelta(days=1)]
#
#     except KeyError:
#         raise KeyError
#
#     roc = ((priceNew - priceOld)/priceOld)*100
#
#     return round(roc,2)
#
#
# def getROCSeries(mydict,window):
#     """
#     Returns a dictionary containing the rate of change(ROC) data for each
#     day.
#
#     @param mydict the dictionary containing the financial data.
#     @param window the window for the ROC to be calculated.
#     @return a dictionary containing ROC data.
#     """
#     rocSeries = {}
#     count = 0
#
#     for k,v in mydict.items():
#         count += 1
#         if count < window+1:
#             rocSeries[datetime.strftime(k,'%Y-%m-%d')] = 0
#             continue
#
#         try:
#             roc = getROC(k,window,mydict)
#             rocSeries[datetime.strftime(k,'%Y-%m-%d')] = roc
#
#         except KeyError:
#             rocSeries[datetime.strftime(k,'%Y-%m-%d')] = 0
#
#     return rocSeries


def getEMA(dateTo,window,data):
    """
    Returns a value for the expontential moving average. This value needs to be
    computed using preious ema's so therefore it runs a simulation. Uses the
    findMovingAverage function to calculate the inital oldEMA. The ema is
    returned as 0 if the data is not available to generate a value.

    @param dateTo the date to find the ema for
    @param window the ema time period.
    @param the dictionary containing the financial data.
    @return an value for the expontential moving average
    """
    multiplier = 2/(window+1)
    count = 0
    newEMA = 0
    for k,v in data.items():
        count += 1
        if count < window:
            continue
        if count == window:
            oldEMA = findMovingAverage(k,window,data)
            continue
        if k == dateTo:
            newEMA = (data[k]*multiplier) + oldEMA * (1-multiplier)
            break
        try:
            newEMA = (data[k]*multiplier) + oldEMA * (1-multiplier)
            oldEMA = newEMA
        except KeyError:
            pass

    return round(newEMA,2)


def getEMASeries(mydict,window):
    """
    Returns a dictionary containing the exponetial moving average(EMA) data for each
    day.

    @param mydict the dictionary containing the financial data.
    @param window the window for the EMA to be calculated.
    @return a dictionary containing EMA data.
    """
    emaSeries = {}
    count = 0

    for k,v in mydict.items():
        count += 1
        if count < window+1:
            emaSeries[datetime.strftime(k,'%Y-%m-%d')] = 0
            continue
        try:
            ema = getEMA(k,window,mydict)
            emaSeries[datetime.strftime(k,'%Y-%m-%d')] = ema

        except OverflowError:
            emaSeries[datetime.strftime(k,'%Y-%m-%d')] = 0

    return emaSeries


def findHL(date,data,window):
    """
    Returns two values containing the highest and lowest price traded from within
    a trading window.

    @param date the date to find the highest and lowest from.
    @param data the dictionary containing the financial data.
    @param window the time period to find the highest and lowest between.
    @return two values. The highest price and the lowest price.
    """
    day = date
    count = 0
    hList = []
    lList = []
    try:
        while count < window: # Going back finding the start date excluding weekends
            try:
                hList.append(data[day][0])
                lList.append(data[day][1])
                count+=1
            except KeyError:
                pass
            day -= timedelta(days=1)

    except KeyError:
        raise KeyError

    high = max(hList)
    low = min(lList)

    return high, low


def getSOSeries(hldata,window):
    """
    Returns two dictionaries containing dates for keys and the Stochastic
    Oscillator as the values. One dictionay contains the keys as string form so
    that a moving average can be found from it. This is to find the %D for the
    stochastic oscillator rule.

    @param data the dictionary containing the financial data of high and low prices.
    @param window the time period to find the highest and lowest between.
    @return two dictionaries.
    """
    soSeries = {}
    soSTRSeries = {}
    count = 0
    for k,v in hldata.items():
        count += 1
        if count < window:
            continue
        p = v[2]
        h,l = findHL(k,hldata,window)
        SO = ((p-l)/(h-l))*100
        soSeries[datetime.strftime(k,'%Y-%m-%d')] = round(SO,2)
        soSTRSeries[k] = round(SO,2)

    return soSeries,soSTRSeries



def getMACDSeries(mydict,slow,fast):
    """
    Returns two dictionaries containing the MACD numbers. Both dictionaires are
    the same, however, macdSeriesDT has the dates in datetime format and
    the dates in macdSeries are in string format. This is because getMacdSignal
    requires the dates to be in datetime format but macdSeries is also needed to
    map the MACD values onto the csv. Must take in a dict with dates in datetime
    format.

    @param mydict the dictionary containing the financial data.
    @param slow the slow ema time period for calculating the MACD values.
    @param fast the fast ema time period for calculating the MACD values.
    @return two dictionaries containing MACD data.
    """
    macdSeries = {}
    macdSeriesDT = {}
    count = 0
    oldMACD = 0
    for k,v in mydict.items():
        count += 1
        if count < 80: # This value could go as low as 26 but its higher to ensure more accurate ema's
            macdSeries[datetime.strftime(k,'%Y-%m-%d')] = 0
            macdSeriesDT[k] = 0
            continue
        try:
            slowEma = getEMA(k,slow,mydict)
            fastEma = getEMA(k,fast,mydict)
            MACD = fastEma - slowEma

            macdSeries[datetime.strftime(k,'%Y-%m-%d')] = round(MACD,2)
            macdSeriesDT[k] = round(MACD,2)

        except OverflowError:
            macdSeries[datetime.strftime(k,'%Y-%m-%d')] = 0
            macdSeriesDT[k] = 0

    return macdSeriesDT, macdSeries


def getMacdSignal(mydict,window):
    """
    Returns a dictionary containing the MACD signal line data. The MACD signal
    line is effectively an ema of the MACD data.

    @param mydict the dictionary containing the MACD data.
    @param window the ema time period for calculating the MACD signal line values.
    @return a dictionary containing MACD signal line data.
    """
    signal = {}
    count = 0
    for k,v in mydict.items():
        count += 1
        if count < 80: #this 80 value should be window + something
            signal[datetime.strftime(k,'%Y-%m-%d')] = 0
            continue

        ema = getEMA(k,window,mydict)
        signal[datetime.strftime( k,'%Y-%m-%d')] = round(ema,2)

    return signal


def getRSI(window,dict1):

    prices = []
    dates = []
    for k,v in dict1.items():
        prices.append(v)
        dates.append(datetime.strftime( k,'%Y-%m-%d'))
    i = 0
    upPrices=[]
    downPrices=[]
    #  Loop to hold up and down price movements
    while i < len(prices):
        if i == 0:
            upPrices.append(0)
            downPrices.append(0)
        else:
            if (prices[i]-prices[i-1])>0:
                upPrices.append(prices[i]-prices[i-1])
                downPrices.append(0)
            elif (prices[i]-prices[i-1])<0:
                downPrices.append(abs(prices[i]-prices[i-1]))
                upPrices.append(0)
            else:
                upPrices.append(0)
                downPrices.append(0)

        i += 1
    x = 0
    avg_gain = []
    avg_loss = []
    #  Loop to calculate the average gain and loss

    while x < len(upPrices):

        if x < window:
            avg_gain.append(0)
            avg_loss.append(0)
        else:
            sumGain = 0
            sumLoss = 0

            y = x-window
            if x == window:
                while y<=x:
                    #print(y)
                    if upPrices[y] == 0:
                        sumLoss += downPrices[y]
                    elif downPrices[y] == 0:
                        sumGain += upPrices[y]

                    y += 1
                avg_gain.append(sumGain/window)
                avg_loss.append(sumLoss/window)
            else:
                avg_gain.append(((avg_gain[x-1]*(window-1)) + upPrices[x])/window)
                avg_loss.append(((avg_loss[x-1]*(window-1)) + downPrices[x])/window)
        x += 1

    p = 0
    RS = []
    RSI = []
    #  Loop to calculate RSI and RS
    while p < len(prices):
        if p < window:
            RS.append(0)
            RSI.append(0)
        else:
            RSvalue = (avg_gain[p]/avg_loss[p])
            RS.append(RSvalue)
            RSI.append(100 - (100/(1+RSvalue)))
        p+=1

    z=0
    rsiDict = {}
    for k,v in dict1.items():
        rsiDict[datetime.strftime( k,'%Y-%m-%d')] = round(RSI[z],2)
        z+=1

    return rsiDict


def trimCSV(csv1,start,end):
    """
    This function trims a csv file to only contain data between certain dates.
    It is used within the load function in order to reduce the file size of the
    csv and improve performance. The original csv file used for precomputaion
    of indicators needs to be larger in order to obtain accurate results. It creates
    a new csv file called Trim.csv which contains the trimmed version of the
    orignial csv

    @param csv1 The name of the larger csv file used for precomputaion.
    @param start the start date of the trimmed csv file
    @param end the end date of the trimmed csv file
    """
    s = datetime.strptime( start,'%Y-%m-%d')
    e = datetime.strptime( end,'%Y-%m-%d')
    dateRange = []

    while(s != e):
        dateRange.append(s)
        s += timedelta(days=1)

    with open(csv1, mode='r') as inp, open('Trim.csv', mode='w') as out:
        writer = csv.writer(out)
        count = 0
        for row in csv.reader(inp):
            if count == 0:
                writer.writerow(row)
                count = 1
                continue
            if datetime.strptime( row[0],'%Y-%m-%d') in dateRange:
                writer.writerow(row)
                continue


def load(ticker,startDate,endDate):
    # download data for 3 years before the startdate to increase accuracy of precomputaion
    beforeStart = datetime.strptime( startDate,'%Y-%m-%d') - timedelta(days=1095)
    yfDownload(ticker,beforeStart,endDate,'1d')
    csv = ticker+".csv"
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


    trimCSV(csv,startDate,endDate)
    os.remove(ticker+'.csv')
    os.rename('Trim.csv',ticker+'.csv')


if __name__ == "__main__":

    load("MSFT",'2017-12-25','2021-02-10')
