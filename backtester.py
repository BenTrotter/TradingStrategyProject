from datetime import datetime, timedelta
import pandas as pd

def movingAveRule(price,dateC,dateP,f,s,panda):

    dates = panda["Date"]
    fast = panda["SMA "+str(f)]
    slow = panda["SMA "+str(s)]

    fastDict = dict(zip(dates, fast))
    slowDict = dict(zip(dates, slow))

    if fastDict[dateP] < slowDict[dateP] and fastDict[dateC] > slowDict[dateC]:
        return 1
    elif fastDict[dateP] > slowDict[dateP] and fastDict[dateC] < slowDict[dateC]:
        return -1
    else:
        return 0

def emaRule(price,dateC,dateP,f,s,panda):

    dates = panda["Date"]
    fast = panda["EMA "+str(f)]
    slow = panda["EMA "+str(s)]

    fastDict = dict(zip(dates, fast))
    slowDict = dict(zip(dates, slow))

    if fastDict[dateP] < slowDict[dateP] and fastDict[dateC] > slowDict[dateC]:
        return 1
    elif fastDict[dateP] > slowDict[dateP] and fastDict[dateC] < slowDict[dateC]:
        return -1
    else:
        return 0

def rsiRule(price,dateC,dateP,h,l,panda):

    dates = panda["Date"]
    rsi = panda["RSI 14"]

    rsiDict = dict(zip(dates, rsi))

    if rsiDict[dateP] < h and rsiDict[dateC] > h:
        return -1
    elif rsiDict[dateP] > l and rsiDict[dateC] < l:
        return 1
    else:
        return 0

def soRule(price,dateC,dateP,h,l,panda):

    dates = panda["Date"]
    so = panda["SO 14"]
    sig = panda["%D 3"]

    soDict = dict(zip(dates, so))
    soSigDict = dict(zip(dates, sig))

    if soDict[dateC] > h and soDict[dateP] > soSigDict[dateP] and soDict[dateC] < soSigDict[dateC]:
        return -1
    elif soDict[dateC] < l and soDict[dateP] < soSigDict[dateP] and soDict[dateC] > soSigDict[dateC]:
        return 1
    else:
        return 0

def simulation(rule):

    f=80
    s=20

    startDate = '2020-01-01'
    endDate = '2020-12-25'
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

        if rule(price,date,oldDate,f,s,panda) == 1 and position == False:
            position = True
            shares = amount/price
            balance -= amount
            print("----- BUY £",amount," -----")
            print('On date: ',date)
            print("Bought ",round(shares,4),' shares at price ',price,'\n')

        elif rule(price,date,oldDate,f,s,panda) == -1 and shares > 0 and position == True:
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

    #print("GETS: ",round(answer,2))

    return round(answer,2), returns+equity

if __name__ == "__main__":
    print(simulation(soRule))
