import random
import operator
import csv
import itertools
from preload import load
import os

from graph import *
from indicators import *

import numpy

import deap
from importlib import reload
reload(deap)

from deap import algorithms, base, creator, tools, gp
from deap.benchmarks.tools import diversity, convergence, hypervolume

from datetime import datetime, timedelta
import pandas as pd

import pickle

from urllib.parse import urlencode
from urllib.request import Request, urlopen

import urllib3
urllib3.disable_warnings()

from scoop import futures

import multiprocessing
from multiprocessing import freeze_support, Pool
from itertools import repeat

import shutil, os


# ------------------------- Global varibales ------------------------- #
file = "MSFT.csv" # needs changing in indicators.py
ticker = "MSFT"
resolution = '1d'

# ---------- Objective Options ---------- #
# 1 : Profit and PC
# 2 : Profit, PC and Risk Exposure
# 3 : Profit, PC, Risk Exposure and number of trades
# 4 : PC and Risk Exposure
# 5 : Profit and Sharpe Ratio
# 6 : Profit, PC and Sharpe Ratio
# 7 : Profit and Risk Exposure
# 8 : Profit and no. trades
# 9 : Profit, Risk Exposure and no. trades
# 10: Profit, Sharpe ratio, Risk Exposure and no.trades
# 11: PC, Sharpe ratio, Risk Exposure and no. trades

objectivesOption = 1

notification = True # True if send a notification when complete

trainingStart = "2018-01-01"
trainingEnd = "2019-01-01"
unseenStart = "2019-01-01"
unseenEnd = "2020-01-01"
validateStart = "2020-01-01"
validateEnd = "2021-01-01"

k = 24
unseenk = 24
riskFreeRate = 0.05
scores = []

# Evolution parameters
ngen = 100
mu = 150
cxpb = 0.4
mutpb = 0.5
# -------------------------------------------------------------------- #

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


# I have forced rsi rule. Maybe allow for more loose options to be explored. e.g.
# not limiting the rsi to be within and if then else rule.

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

emaWindows = [5, 12, 26, 30, 50, 100, 200]
for i in emaWindows:
    pset.addTerminal(i,ema_int)

maWindows = [5, 10, 20, 30, 50, 100, 200]
for i in maWindows:
    pset.addTerminal(i,ma_int)

rsiWindows = [7, 14, 28]
for i in rsiWindows:
    pset.addTerminal(i,rsi_int)
for i in range(0,100):
    pset.addTerminal(i,rsi_bounds)

macdS = [26, 35]
for i in macdS:
    pset.addTerminal(i,macd_s)
macdF = [5, 12]
for i in macdF:
    pset.addTerminal(i,macd_f)
macdSig = [5, 9]
for i in macdSig:
    pset.addTerminal(i,macd_sig)
pset.addTerminal(0,macd_bounds)

soWindows = [7,14,28]
for i in soWindows:
    pset.addTerminal(i,so_int)
pset.addTerminal(0,so_bounds)

pset.renameArguments(ARG0='Date')
pset.renameArguments(ARG1='Position')

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
    gets a dict of date and price as keys and values. It trims the dict
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

def simulation(individual):

    rule = toolbox.compile(expr=individual)

    startDate = trainingStart # Start date of the overall trading window.
    endDate = trainingEnd # End date of the overall trading window.
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
    pcSplit = k # The number of intervals to split the trading window into for PC
    # Get the interval in days for PC
    interval = splitTrainingPeriod(startDay, endDay, pcSplit)
    # Get the dates that the PC count needs to be checked and updated.
    performanceConsistencyDates = getPCUpdateDates(startDay, endDay, interval)
    # Gets a dict of date and price as keys and values.
    priceData = getPriceDataDict(file,startDate,endDate)
    riskExposure = 0 # The num days that a trader is in the market.
    pcCount = 0 # The count for the performancy consistency value.
    pcIter = 0 # The index for the performanceConsistencyDatess
    answer = 1 # Initialising answer to prevent zero division error.

    dailyReturn = []
    oldP = False
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
            percentIncPriceTest = ((price-oldPrice)/oldPrice)*100
            percentIncBalanceStrategy =  ((balance-oldBalance)/oldBalance)*100
            if percentIncPriceTest <= percentIncBalanceStrategy:
                pcCount += 1
            oldPrice = price
            oldBalance = balance
            pcIter+=1

        action = rule(date,position)

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

        elif sell and shares > 0:
            position = False
            balance = round(shares*price, 2)
            profit = balance - oldAmount
            shares = 0

        elif shares != 0:
            balance = round(price*shares,2)

        if position == True:
            riskExposure += 1

        # Ends the simulation at the end date
        if datetime.strptime(date,'%Y-%m-%d') >= endDay:
            bhBalance = bhShares*priceData[oldDate]
            break

        oldDate = date

        if oldP != False:
            if position:
                # dailyReturn.append((((price-oldP)/oldP)*100)-riskFreeRate/numTDays) # 252 annualises the ratio
                dailyReturn.append(((price-oldP)/oldP)*100)
            # else:
            #     # dailyReturn.append(0-riskFreeRate/numTDays)
            #     dailyReturn.append(0)

        oldP = price

    # Final peformance consitency check on final iterval
    if(pcIter != pcSplit):
        percentIncPriceTest = ((price-oldPrice)/oldPrice)*100
        percentIncBalanceStrategy =  ((balance-oldBalance)/oldBalance)*100
        if percentIncPriceTest <= percentIncBalanceStrategy:
            pcCount += 1
        pcIter+=1

    # Check to ensure every interval as been included in the PC count
    if(pcIter != pcSplit):
        print("\nASSERT ERROR: Not all pc intervals have been calculated.\n")

    # Percentage return
    answer = ((balance - startingBalance)/startingBalance)*100

    # Sharpe Ratio
    if len(dailyReturn) == 0:
        sharpeRatio = 0
    else:
        aveDailyReturn = sum(dailyReturn)/len(dailyReturn)
        stdDailyRateOfReturn = numpy.std(dailyReturn)
        # sharpeRatio = round(numpy.sqrt(numTDays) * aveDailyReturn / stdDailyRateOfReturn,2)
        if stdDailyRateOfReturn == 0:
            sharpeRatio = 0
        else:
            sharpeRatio = round((aveDailyReturn-(riskFreeRate/numTDays))/stdDailyRateOfReturn,2)


    if numTrades == 0:
        numTrades = 100
        riskExposure = 100

    if objectivesOption == 1:
        return round(answer,2), pcCount,
    elif objectivesOption == 2:
        return round(answer,2), pcCount, riskExposure,#numTrades,
    elif objectivesOption == 3:
        return round(answer,2), pcCount, riskExposure, numTrades,
    elif objectivesOption == 4:
        return pcCount, riskExposure,
    elif objectivesOption == 5:
        return round(answer,2), sharpeRatio,
    elif objectivesOption == 6:
        return round(answer,2), pcCount, sharpeRatio,
    elif objectivesOption == 7:
        return round(answer,2), riskExposure,
    elif objectivesOption == 8:
        return round(answer,2), numTrades,
    elif objectivesOption == 9:
        return round(answer,2), riskExposure, numTrades,
    elif objectivesOption == 10:
        return round(answer,2), sharpeRatio, riskExposure, numTrades,
    elif objectivesOption == 11:
        return pcCount, sharpeRatio,riskExposure, numTrades,

if objectivesOption == 1:
    creator.create("Fitness", base.Fitness, weights=(1.0,1.0))
elif objectivesOption == 2:
    creator.create("Fitness", base.Fitness, weights=(1.0,1.0,-1.0))
elif objectivesOption == 3:
    creator.create("Fitness", base.Fitness, weights=(1.0,1.0,-1.0,-1.0))
elif objectivesOption == 4:
    creator.create("Fitness", base.Fitness, weights=(1.0,-1.0))
elif objectivesOption == 5:
    creator.create("Fitness", base.Fitness, weights=(1.0,1.0))
elif objectivesOption == 6:
    creator.create("Fitness", base.Fitness, weights=(1.0,1.0,1.0))
elif objectivesOption == 7:
    creator.create("Fitness", base.Fitness, weights=(1.0,-1.0))
elif objectivesOption == 8:
    creator.create("Fitness", base.Fitness, weights=(1.0,-1.0))
elif objectivesOption == 9:
    creator.create("Fitness", base.Fitness, weights=(1.0,-1.0,-1.0))
elif objectivesOption == 10:
    creator.create("Fitness", base.Fitness, weights=(1.0,1.0,-1.0,-1.0))
elif objectivesOption == 11:
    creator.create("Fitness", base.Fitness, weights=(1.0,1.0,-1.0,-1.0))

creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

terminal_types = [so_bounds,so_int,macd_bounds,macd_sig,macd_f,macd_s, ema_int ,ma_int, rsi_int, rsi_bounds, str, position_bool]
toolbox = base.Toolbox()

toolbox.register("expr", gp.generate_safe, pset=pset, min_=1, max_=5, terminal_types=terminal_types)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", simulation)
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.generate_safe, min_=1, max_=5, terminal_types=terminal_types)
# Check mutation is working properly
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)



def main(s,e,parallel=True,save=True):

    random.seed()

    trainBH = findBH(file,s,e)

    NGEN = ngen
    MU = mu
    CXPB = cxpb
    MUTPB = mutpb
    print("\n * ----------------- Evolution Info ----------------- *")
    if objectivesOption == 1:
        print("Using two objectives: Profit and PC")
    elif objectivesOption == 2:
        print("Using three objectives: Profit, PC and Risk Exposure")
    elif objectivesOption == 3:
        print("Using four objectives: Profit, PC, Risk Exposure and Number of trades")
    elif objectivesOption == 4:
        print("Using two objectives: PC and Risk Exposure")
    elif objectivesOption == 5:
        print("Using two objectives: Profit and Sharpe Ratio")
    elif objectivesOption == 6:
        print("Using three objectives: Profit, PC and Sharpe Ratio")
    elif objectivesOption == 7:
        print("Using two objectives: Profit and Risk Exposure")
    elif objectivesOption == 8:
        print("Using two objectives: Profit and Number of trades")
    elif objectivesOption == 9:
        print("Using three objectives: Profit, Risk Exposure and Number of trades")
    elif objectivesOption == 10:
        print("Using four objectives: Profit, Sharpe Ratio, Risk Exposure and Number of trades")
    elif objectivesOption == 11:
        print("Using four objectives: PC, Sharpe Ratio, Risk Exposure and Number of Trades")
    print("Retrieving data from ", file)
    print("Number of generations: ",NGEN)
    print("Population size: ",MU)
    print("CXPB: ",CXPB)
    print("MUTPB: ",MUTPB)
    print("PC k value is: ",k)
    print("Training PC split is: ",splitTrainingPeriod(datetime.strptime(s,'%Y-%m-%d'), datetime.strptime(e,'%Y-%m-%d'), k),"\n")
    print("Training B&H is ",trainBH,"%")
    print("Training on data from ",s," to ",e,"\n")

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "avg", "max"

    #multiprocessing
    if parallel:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
        toolbox.register("map", futures.map)

    pop = toolbox.population(n=MU)

    paretofront = tools.ParetoFront()
    all = []
    hypers = {}

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)
        paretofront.update(pop)
        for ind in pop:
            if ind not in all:
                all.append(ind)

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

        if objectivesOption == 1:
            hypers[gen] = hypervolume(pop, [0.0, 0.0])
        elif objectivesOption == 2:
            hypers[gen] = hypervolume(pop, [0.0, 0.0, 500])
        elif objectivesOption == 3:
            hypers[gen] = hypervolume(pop, [0.0, 0.0, 500, 500])
        elif objectivesOption == 4:
            hypers[gen] = hypervolume(pop, [0.0, 500])
        elif objectivesOption == 5:
            hypers[gen] = hypervolume(pop, [0.0, 0.0])
        elif objectivesOption == 6:
            hypers[gen] = hypervolume(pop, [0.0, 0.0, 0.0])
        elif objectivesOption == 7:
            hypers[gen] = hypervolume(pop, [0.0, 500])
        elif objectivesOption == 8:
            hypers[gen] = hypervolume(pop, [0.0, 500])
        elif objectivesOption == 9:
            hypers[gen] = hypervolume(pop, [0.0, 500, 500])
        elif objectivesOption == 10:
            hypers[gen] = hypervolume(pop, [0.0, 0.0, 500, 500])
        elif objectivesOption == 11:
            hypers[gen] = hypervolume(pop, [0.0, 0.0, 500, 500])

        print(logbook.stream)

    if save:
        cp = dict(population=pop, generation=gen, pareto=paretofront,
            logbook=logbook, all=all, rndstate=random.getstate(),
            k=k,mu=MU,cxpb=CXPB,mutpb=MUTPB,hypers=hypers)

        with open("SavedOutput.pkl", "wb") as cp_file:
            pickle.dump(cp, cp_file)

    if parallel:
        pool.close()

    if notification:
        url = 'https://www.pushsafer.com/api' # Set destination URL here
        post_fields = {                       # Set POST fields here
        	"t" : "Trading Strategy Evolution Complete",
        	"m" : "Please attend your laptop to sort and evaluate your data.",
        	"s" : "1",
        	"v" : "2",
        	"i" : "",
        	"c" : "",
        	"d" : "",
        	"u" : "",
        	"ut" : "",
        	"k" : "ar8KrxDHmzlniBs6MUlf"
        	}

        request = Request(url, urlencode(post_fields).encode())
        json = urlopen(request).read().decode()

    if objectivesOption == 1:
        plot_pop_pareto_front(all, paretofront, objectivesOption, "Fitness of all individuals")
    elif objectivesOption == 2:
        allValues = []
        for i in all:
            allValues.append(i.fitness.values)
        scatter(all,paretofront,objectivesOption)
        threeDimensionalPlot(allValues,dominates,objectivesOption)
    elif objectivesOption == 3:
        scatter(all,paretofront,objectivesOption)
    elif objectivesOption == 4:
        plot_pop_pareto_front(all, paretofront,objectivesOption, "Fitness of all individuals")
    elif objectivesOption == 5:
        plot_pop_pareto_front(all, paretofront, objectivesOption, "Fitness of all individuals")
    elif objectivesOption == 6:
        allValues = []
        for i in all:
            allValues.append(i.fitness.values)
        scatter(all,paretofront,objectivesOption)
        threeDimensionalPlot(allValues,dominates,objectivesOption)
    elif objectivesOption == 7:
        plot_pop_pareto_front(all, paretofront, objectivesOption, "Fitness of all individuals")
    elif objectivesOption == 8:
        plot_pop_pareto_front(all, paretofront, objectivesOption, "Fitness of all individuals")
    elif objectivesOption == 9:
        allValues = []
        for i in all:
            allValues.append(i.fitness.values)
        scatter(all,paretofront,objectivesOption)
        threeDimensionalPlot(allValues,dominates,objectivesOption)
    elif objectivesOption == 10:
        scatter(all,paretofront,objectivesOption)
    elif objectivesOption == 11:
        scatter(all,paretofront,objectivesOption)

    plot_hypervolume(hypers)

    return paretofront, logbook

def findBH(name,tStart,tEnd):

    # Gets a dict of date and price as keys and values.
    priceData = getPriceDataDict(name,tStart,tEnd)
    dataList = list(priceData.values())
    p1 = dataList[0]
    p2 = dataList[-1]

    bh = round(((p2-p1)/p1)*100,2)

    return bh

def paraUnseen(i,unseenBH,data):

    if data == "unseen":
        if i.fitness.values[0] in scores:
            return
        scores.append(i.fitness.values[0])

    rule = toolbox.compile(expr=i)

    if data == "unseen":
        startDate = unseenStart # Start date of the overall trading window.
        endDate = unseenEnd # End date of the overall trading window.
    elif data == "validation":
        startDate = validateStart# Start date of the overall trading window.
        endDate = validateEnd # End date of the overall trading window.

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
    pcSplit = unseenk# The number of intervals to split the trading window into for PC
    # Get the interval in days for PC
    interval = splitTrainingPeriod(startDay, endDay, pcSplit)
    # Get the dates that the PC count needs to be checked and updated.
    performanceConsistencyDates = getPCUpdateDates(startDay, endDay, interval)
    # Gets a dict of date and price as keys and values.
    priceData = getPriceDataDict(file,startDate,endDate)
    riskExposure = 0 # The num days that a trader is in the market.
    pcCount = 0 # The count for the performancy consistency value.
    pcIter = 0 # The index for the performanceConsistencyDatess
    answer = 1 # Initialising answer to prevent zero division error.

    dailyReturn = []
    oldP = False
    numTrDays = len(priceData)

    for date, price in priceData.items(): # Need to make within the sim time frame
        # to start the sim from the start date
        if datetime.strptime(date,'%Y-%m-%d') < startDay and not findStart:
            continue
        # calculating the b&h strategy at start date and the first PC interval.
        elif not findStart:
            startD = date
            findStart = True
            oldPrice = price
            oldBalance = startingBalance

        if iter == 0:
            oldDate = date
            iter += 1
            continue

        # PC update if the correct date has been reached
        if datetime.strptime(date, '%Y-%m-%d') >= performanceConsistencyDates[pcIter]:
            percentIncPriceTest = ((price-oldPrice)/oldPrice)*100
            percentIncBalanceStrategy =  ((balance-oldBalance)/oldBalance)*100
            if percentIncPriceTest <= percentIncBalanceStrategy:
                pcCount += 1
            oldPrice = price
            oldBalance = balance
            pcIter+=1

        action = rule(date,position)

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

        elif sell and shares > 0:
            position = False
            balance = round(shares*price, 2)
            profit = balance - oldAmount
            shares = 0

        elif shares != 0:
            balance = round(price*shares,2)

        if position == True:
            riskExposure += 1

        # Ends the simulation at the end date
        if datetime.strptime(date,'%Y-%m-%d') >= endDay:
            break

        oldDate = date

        if oldP != False:
            if position:
                # dailyReturn.append((((price-oldP)/oldP)*100)-riskFreeRate/numTrDays)
                dailyReturn.append((((price-oldP)/oldP)*100))
            # else:
            #     # dailyReturn.append(0-riskFreeRate/numTrDays) # add risk free rate of 0.05 as a global variable
            #     dailyReturn.append(0)

        oldP = price

    # Final peformance consitency check on final iterval
    if(pcIter != pcSplit):
        percentIncPriceTest = ((price-oldPrice)/oldPrice)*100
        percentIncBalanceStrategy =  ((balance-oldBalance)/oldBalance)*100
        if percentIncPriceTest <= percentIncBalanceStrategy:
            pcCount += 1
        pcIter+=1

    # Check to ensure every interval as been included in the PC count
    if(pcIter != pcSplit):
        print("\nASSERT ERROR: Not all pc intervals have been calculated.\n")

    answer = ((balance - startingBalance)/startingBalance)*100

    if len(dailyReturn) == 0:
        sharpe = 0
    else:
        aveDailyReturn = sum(dailyReturn)/len(dailyReturn)
        numTDays = len(dailyReturn)
        stdDailyRateOfReturn = numpy.std(dailyReturn)
        if stdDailyRateOfReturn == 0:
            sharpe = 0
        else:
            # sharpe = round(numpy.sqrt(numTrDays) * aveDailyReturn / stdDailyRateOfReturn,2)
            sharpe = round((aveDailyReturn-(riskFreeRate/numTDays))/stdDailyRateOfReturn,2)

    if numTrades == 0:
        numTrades = 100
        riskExposure = 100

    above = round(((round(answer,2)-round(unseenBH,2))/round(unseenBH,2))*100,2)
    # print("Training score: ",i.fitness.values[0]," Unseen score: ",round(answer,2),'\n')
    # pcDict[str(i)] = [pcCount,round(answer,2),i.fitness.values[0],sharpe,above]
    if data == "unseen":
        return [str(i),pcCount,round(answer,2),i.fitness.values[0],sharpe,above]
    elif data == "validation":
        return [str(i),pcCount,round(answer,2),0,sharpe,above]

def unseen(paretofront, tStart, tEnd,test_K, fileName, data):

    BandH = True
    bh = findBH(fileName,tStart,tEnd)
    if data == "unseen":
        print("Unseen B&H is ", bh,"%")
        print("Testing on unseen data from ",tStart," to ", tEnd,'\n')
    elif data == "validation":
        print("Validation B&H is ", bh,"%")
        print("Testing on validation data from ",tStart," to ", tEnd,'\n')

    pcDict = {}
    pcSplit = unseenk
    startDay = datetime.strptime(tStart,'%Y-%m-%d')
    endDay = datetime.strptime(tEnd,'%Y-%m-%d')
    interval = splitTrainingPeriod(startDay, endDay, pcSplit)

    if data == "unseen":
        print("Number on pareto front is ",len(paretofront))
        print("Using ",multiprocessing.cpu_count()," processors. ")
    elif data == "validation":
        print("First result is best profit on pareto, second is best PC.\n")

    if data == "unseen":
        with Pool() as p:
            mapped = p.starmap(paraUnseen, zip(paretofront, repeat(bh), repeat('unseen')))
    elif data == "validation":
        a = paraUnseen(paretofront[0], bh, "validation")
        b = paraUnseen(paretofront[1], bh, "validation")
        mapped = [a,b]


    for i in mapped:
        if i is not None:
            pcDict[i[0]] = i[1:]

    cp = dict(pareto=pcDict)
    with open("FinalOutput.pkl", "wb") as cp_file:
        pickle.dump(cp, cp_file)

    return pcDict, interval


def processPareto(paretoDict, interval):

    print("Interval length for unseen PC is: ",interval,"\n")

    # Ordering the dictionary by pc value
    sorted_d = dict( sorted(paretoDict.items(), key=operator.itemgetter(1),reverse=True))

    beating = []
    maxProfit = -100
    maxPC = 0
    count = 0
    for key,v in sorted_d.items():
        if count == 0:
            pcKey = key
            count = 1
        print("Strategy:")
        print(key)
        print("Achieved a pc score of ",v[0],"/",unseenk, " on unseen data.")
        print("Training score: ",v[2])
        print("Unseen score: ",v[1]," -> This is an change of ",v[4],"% from the B&H.")
        print("Sharpe ratio: ",v[3],'\n')
        if v[1] > maxProfit:
            maxProfit = v[1]
            percent = v[4]
            pcMax = v[0]
            profKey = key
            
        if v[4] > 0:
            beating.append(v)
    
    print('\nMaximum profit is ', maxProfit," -> This is an change of ",percent,"% from the B&H.")
    print('PC value for this was ',pcMax,'/', unseenk)
    print('\nNumber that outperformed BH is ',len(beating),'/',len(sorted_d),'\n')

    bestOnPareto = [profKey, pcKey]

    return bestOnPareto


if __name__ == "__main__":

    # Load training data
    load(ticker,trainingStart,trainingEnd,resolution)
    freeze_support()
    random.seed()
    # Evolution on training data
    pareto, stats = main(trainingStart,trainingEnd)

    # Load Unseen data
    load(ticker,unseenStart,unseenEnd,resolution)
    # Run pareto front solutions on unseen data
    answer, interval = unseen(pareto, unseenStart, unseenEnd, unseenk,file,"unseen")
    # Retrieve unseen results and process/print results
    bestPareto = processPareto(answer,interval)

    # Load Validation data
    load(ticker,validateStart,validateEnd,resolution)
    # Run pareto front solutions on validation data
    vAnswer, vInterval = unseen(bestPareto, validateStart, validateEnd, unseenk,file,"validation")
    bestPareto = processPareto(vAnswer,vInterval)

    # Remove old results folder
    try:
        cwd = os.getcwd()
        shutil.rmtree(cwd+"/results")
    except FileNotFoundError:
        print("No file name results to remove. Continuing...")
    os.mkdir("results")
    # Move final output files into a separate folder
    files = ['FinalOutput.pkl', 'SavedOutput.pkl', 'hv.png','map.png','3d.png','scatter.png']
    for f in files:
        try:
            shutil.move(f, 'results')
        except:
            continue
