import random
import operator
import csv
import itertools
from preload import load

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
from multiprocessing import freeze_support

# ------------------------- Global varibales ------------------------- #
file = "MSFT.csv"
ticker = "MSFT"
resolution = '1d'

# ---------- Objective Options ---------- #
# 1 : Profit and PC
# 2 : Profit, PC and Risk Exposure
# 3 : Profit, PC, Risk Exposure and number of trades
# 4 : PC and Risk Exposure
# 5 : Profit and Sharpe Ratio
# 6 : Profit, PC and Sharpe Ratio
objectivesOption = 2

notification = False # True if send a notification when complete

trainingStart = "2013-01-01"
trainingEnd = "2015-01-01"
unseenStart = "2015-01-01"
unseenEnd = "2021-01-01"
k = 12
unseenk = 6
riskFreeRate = 0.05

# Evolution parameters
ngen = 80
mu = 100
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

def ma(date,window):

    panda = pd.read_csv(file)
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
                dailyReturn.append((((price-oldP)/oldP)*100)-riskFreeRate/numTDays) # 252 annualises the ratio
            else:
                dailyReturn.append(0-riskFreeRate/numTDays)

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
    aveDailyReturn = sum(dailyReturn)/len(dailyReturn)
    stdDailyRateOfReturn = numpy.std(dailyReturn)
    sharpeRatio = round(numpy.sqrt(numTDays) * aveDailyReturn / stdDailyRateOfReturn,2)

    # Buy and hold calculation.
    # bhEndValue = bhShares * price
    # bhIncrease = ((bhEndValue-startingBalance)/startingBalance)*100
    # print("Buy and Hold % increase for seen is ", round(bhIncrease,2),"\n")

    # Sharpe Ratio

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




def plot_pop_pareto_front(pop,paretofront, option, title=""):
    x=[]
    y=[]
    for p in paretofront:
        fitness = p.fitness.values
        x.append(fitness[0])
        y.append(fitness[1])
    xp=[]
    yp=[]
    for p in pop:
        fitness = p.fitness.values
        xp.append(fitness[0])
        yp.append(fitness[1])
    fig,ax=plt.subplots(figsize=(5,5))
    ax.plot(xp,yp,".", label="Population")
    ax.plot(x,y,".", label="Pareto Front")
    fitpareto=list(zip(x,y))
    fitpop=list(zip(xp,yp))
    print("\nPareto Front: "+str(fitpareto),'\n')
    # print("Population: "+str(fitpop))

    ax.set_title(title)
    if option == 1:
        plt.xlabel('% increase in profit')
        plt.ylabel('Performance consitency')
    if option == 4:
        plt.xlabel('Performance consistency')
        plt.ylabel('Risk Exposure')
    if option == 5:
        plt.xlabel('% increase in profit')
        plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.savefig('map.png')

def plot_hypervolume(hypers, title="Hypervolume"):
    x=[]
    y=[]
    for gen,hv in hypers.items():
        x.append(gen)
        y.append(hv)

    fig,ax=plt.subplots(figsize=(5,5))
    fitpareto=list(zip(x,y))

    ax.set_title(title)
    plt.plot(x, y, color='red', marker='o')
    plt.xlabel('Generation Number')
    plt.ylabel('Hypervolume')
    plt.savefig('hv.png')

def threeScatterPlot(allValues,option):
    if option == 2:
        df = pd.DataFrame(allValues, columns=['Profit','PC','Risk Exposure'])
    elif option == 3:
        df = pd.DataFrame(allValues, columns=['Profit','PC','Risk Exposure','No. Trades'])
    elif option == 6:
        df = pd.DataFrame(allValues, columns=['Profit','PC','Sharpe Ratio'])

    scatter = pd.plotting.scatter_matrix(df, alpha=0.2)
    plt.savefig(r"scatter.png")


def threeDimensionalPlot(inputPoints, dominates, option):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if option == 2:
        ax.set_xlabel('% increase in profit', rotation=150)
        ax.set_ylabel('Performance consitency')
        ax.set_zlabel('Risk Exposure', rotation=60)
    elif option == 6:
        ax.set_xlabel('% increase in profit', rotation=150)
        ax.set_ylabel('Performance consitency')
        ax.set_zlabel('Sharpe Ratio', rotation=60)
    dp = numpy.array(list(dominatedPoints))
    pp = numpy.array(list(paretoPoints))
    print(pp.shape,dp.shape)
    ax.scatter(dp[:,0],dp[:,1],dp[:,2])
    ax.scatter(pp[:,0],pp[:,1],pp[:,2],color='red')

    import matplotlib.tri as mtri
    triang = mtri.Triangulation(pp[:,0],pp[:,1])
    ax.plot_trisurf(triang,pp[:,2],color='red')
    plt.savefig('3d.png')

    return paretoPoints, dominatedPoints

def dominates(row, candidateRow):
    return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row)


def main(s,e,parallel=True,save=True):

    random.seed()

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
    print("Retrieving data from ", file)
    print("Number of generations: ",NGEN)
    print("Population size: ",MU)
    print("CXPB: ",CXPB)
    print("MUTPB: ",MUTPB)
    print("PC k value is: ",k)
    print("Training PC split is: ",splitTrainingPeriod(datetime.strptime(s,'%Y-%m-%d'), datetime.strptime(e,'%Y-%m-%d'), k),"\n")
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
        # Vary the population
        # offspring = tools.selTournamentDCD(pop, len(pop))
        # offspring = [toolbox.clone(ind) for ind in offspring]
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)


        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)

        #
        #
        # for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        #     if random.random() <= CXPB:
        #         toolbox.mate(ind1, ind2)
        #
        #     toolbox.mutate(ind1)
        #     toolbox.mutate(ind2)
        #     del ind1.fitness.values, ind2.fitness.values

        # # Evaluate the individuals with an invalid fitness
        # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        # for ind, fit in zip(invalid_ind, fitnesses):
        #     ind.fitness.values = fit

        # Select the next generation population
        # pop = toolbox.select(pop + offspring, MU)
        # print("Length: ",len(pop))
        # for i in pop:
        #     print(i)
        # print('\n')
        # check to ensure this update of paretofront is in the right place
        paretofront.update(pop)
        for ind in pop:
            if ind not in all:
                all.append(ind)

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

        if objectivesOption == 1:
            hypers[gen] = hypervolume(pop, [1.0, 1.0])
        elif objectivesOption == 2:
            hypers[gen] = hypervolume(pop, [1.0, 1.0, 50])
        elif objectivesOption == 3:
            hypers[gen] = hypervolume(pop, [1.0, 1.0, 50, 200])
        elif objectivesOption == 4:
            hypers[gen] = hypervolume(pop, [1.0, 50])
        elif objectivesOption == 5:
            hypers[gen] = hypervolume(pop, [1.0, 0.5])
        elif objectivesOption == 6:
            hypers[gen] = hypervolume(pop, [1.0, 1.0, 0.5])

        print(logbook.stream)

    if save:
        cp = dict(population=pop, generation=gen, pareto=paretofront,
            logbook=logbook, all=all, rndstate=random.getstate())

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
        threeScatterPlot(allValues,objectivesOption)
        threeDimensionalPlot(allValues,dominates,objectivesOption)
    elif objectivesOption == 3:
        allValues = []
        for i in all:
            allValues.append(i.fitness.values)
        threeScatterPlot(allValues,objectivesOption)
    elif objectivesOption == 4:
        plot_pop_pareto_front(all, paretofront,objectivesOption, "Fitness of all individuals")
    elif objectivesOption == 5:
        plot_pop_pareto_front(all, paretofront, objectivesOption, "Fitness of all individuals")
    elif objectivesOption == 6:
        allValues = []
        for i in all:
            allValues.append(i.fitness.values)
        threeScatterPlot(allValues,objectivesOption)
        threeDimensionalPlot(allValues,dominates,objectivesOption)

    plot_hypervolume(hypers)


    #print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0, 11.0]))

    return paretofront, logbook




def unseen(paretofront, tStart, tEnd):

    BandH = True
    print("\nTesting on unseen data from ",tStart," to ", tEnd)
    pcDict = {}

    print("Number on pareto front is ",len(paretofront))
    count3 = 1

    for i in paretofront:
        print(count3)
        count3 += 1

        rule = toolbox.compile(expr=i)

        startDate = tStart # Start date of the overall trading window.
        endDate = tEnd # End date of the overall trading window.
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
        pcSplit = unseenk # The number of intervals to split the trading window into for PC
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
                    dailyReturn.append((((price-oldP)/oldP)*100)-riskFreeRate/numTrDays)
                else:
                    dailyReturn.append(0-riskFreeRate/numTrDays) # add risk free rate of 0.05 as a global variable

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

        if BandH:
            bhEndValue = bhShares * price
            bhIncrease = ((bhEndValue-startingBalance)/startingBalance)*100

        answer = ((balance - startingBalance)/startingBalance)*100

        aveDailyReturn = sum(dailyReturn)/len(dailyReturn)
        stdDailyRateOfReturn = numpy.std(dailyReturn)
        sharpe = round(numpy.sqrt(numTrDays) * aveDailyReturn / stdDailyRateOfReturn,2)

        if BandH:
            print("Buy and Hold % increase for unseen is ", round(bhIncrease,2),"\n")
            BandH = False

        if numTrades == 0:
            numTrades = 100
            riskExposure = 100

        above = round(((round(answer,2)-round(bhIncrease,2))/round(bhIncrease,2))*100,2)
        # print(i)
        # print("Training score: ",i.fitness.values[0]," Unseen score: ",round(answer,2),'\n')
        pcDict[str(i)] = [pcCount,round(answer,2),i.fitness.values[0],sharpe,above]

    return pcDict, interval

def processPareto(paretoDict, interval):

    print("Interval length for unseen PC is: ",interval,"\n")

    # Ordering the dictionary by pc value
    sorted_d = dict( sorted(paretoDict.items(), key=operator.itemgetter(1),reverse=True))

    for key,v in sorted_d.items():
        print("Strategy:")
        print(key)
        print("Achieved a pc score of ",v[0],"/",unseenk, " on unseen data.")
        print("Training score: ",v[2])
        print("Unseen score: ",v[1]," -> This is an change of ",v[4],"% from the B&H.")
        print("Sharpe ratio: ",v[3],'\n')

# Graph of final unseen results taking just the top ten maybe
# def graphUnseenResults(paretoDict):

    #

if __name__ == "__main__":
    load(ticker,trainingStart,unseenEnd,resolution)
    freeze_support()
    random.seed()
    pareto, stats = main(trainingStart,trainingEnd)
    answer, interval = unseen(pareto, unseenStart, unseenEnd)
    processPareto(answer,interval)
