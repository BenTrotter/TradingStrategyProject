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

import array
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume







class pd_float(object):
    pass

class pd_bool(object):
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

# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", [str], pd_bool)
# boolean operators
pset.addPrimitive(operator.and_, [pd_bool, pd_bool], pd_bool)
pset.addPrimitive(operator.or_, [pd_bool, pd_bool], pd_bool)
pset.addPrimitive(operator.not_, [pd_bool], pd_bool)
pset.addPrimitive(operator.lt, [pd_float, pd_float], pd_bool)
pset.addPrimitive(operator.gt, [pd_float, pd_float], pd_bool)
pset.addPrimitive(operator.lt, [rsi_bounds, rsi_ans], pd_bool)
pset.addPrimitive(operator.gt, [rsi_bounds, rsi_ans], pd_bool)
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


def simulation(individual):

    rule = toolbox.compile(expr=individual)

    startDate = '2020-01-01'
    endDate = '2020-12-25'
    amount = 1000
    shares = 0
    balance = 1000
    returns = 0
    equity = 0
    numTrades = 1
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


        action = rule(date)

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
            numTrades += 1
            # print("----- BUY £",amount," -----")
            # print('On date: ',date)
            # print("Bought ",round(shares,4),' shares at price ',price,'\n')

        elif sell and shares > 0 and position == True:
            position = False
            balance = shares*price
            profit = balance - amount
            returns += profit
            # print("----- SELL £",round(balance,2)," -----")
            # print("Profit from indivdual sale: ",round(profit,2))
            # print('On date: ',date)
            # print("Sold ",round(shares,4),' shares at price ',price,'\n')

        elif shares != 0:
            equity = price*shares

        # to end the sim at the end date
        if datetime.strptime(date,'%Y-%m-%d') >= endDay:
            bhBalance = bhShares*combined[oldDate]
            break

        oldDate = date
        answer = (((returns+equity)-amount)/amount)*100

    return round(answer,2), numTrades#, returns+equity






creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

terminal_types = [so_bounds,so_int,macd_bounds,macd_sig,macd_f,macd_s, ema_int ,ma_int, rsi_int, rsi_bounds, str]

toolbox = base.Toolbox()

toolbox.register("expr", gp.generate_safe, pset=pset, min_=1, max_=7, terminal_types=terminal_types)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", simulation)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.generate_safe, min_=1, max_=5, terminal_types=terminal_types)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset) # I'm worried that mutation isn't working properly
toolbox.register("select", tools.selNSGA2)

def main(seed=None):
    random.seed(seed)

    NGEN = 5
    MU = 10
    CXPB = 0.8

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

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
        offspring = tools.selTournamentDCD(pop, 8) #i added 8
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

    return pop, logbook

if __name__ == "__main__":
    # with open("pareto_front/zdt1_front.json") as optimal_front_data:
    #     optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    # optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))

    pop, stats = main()
    pop.sort(key=lambda x: x.fitness.values)
