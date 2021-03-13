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

from deap import algorithms, base, creator, tools, gp
from deap.benchmarks.tools import diversity, convergence, hypervolume

from datetime import datetime, timedelta
import pandas as pd

from scoop import futures

import multiprocessing
from multiprocessing import freeze_support

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

    panda = pd.read_csv('MSFT.csv') # Maybe set MSFT.csv variable to a global variable
    mas = panda["SMA "+str(window)+".0"]
    dates = panda["Date"]
    dict1 = dict(zip(dates,mas))
    ma = dict1[date]
    return ma

def rsi(date,window):

    panda = pd.read_csv('MSFT.csv')
    rsi = panda["RSI "+str(window)+".0"]
    dates = panda["Date"]
    dict1 = dict(zip(dates,rsi))
    RSI = dict1[date]
    return RSI

def ema(date,window):

    panda = pd.read_csv('MSFT.csv')
    emas = panda["EMA "+str(window)+".0"]
    dates = panda["Date"]
    dict1 = dict(zip(dates,emas))
    ema = dict1[date]
    return ema

def macd(date,s,f,sig):

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

    startDate = '2020-01-01'
    endDate = '2021-01-01'
    amount = 1000
    shares = 0
    balance = 1000
    returns = 0
    equity = 0
    startDay = datetime.strptime(startDate,'%Y-%m-%d')
    endDay = datetime.strptime(endDate,'%Y-%m-%d')

    priceData = getPriceDataDict('MSFT.csv',startDate,endDate)

    iter = 0
    startCount = 0
    bhBalance = 0
    bhShares = 0
    numTrades = 0
    position = False

    for date, price in priceData.items(): # Need to make within the sim time frame
        # to start the sim from the start date
        if datetime.strptime(date,'%Y-%m-%d') < startDay and startCount == 0:
            continue
        # calculating the b&h strategy at start date
        elif startCount == 0:
            startD = date
            startCount = 1
            bhShares = amount / priceData[date]

        if iter == 0:
            oldDate = date
            iter += 1
            continue

        action = rule(date,position)

        if action and position == False:
            buy = True
            sell = False
        elif not action and position == False:
            sell = False
            buy = False
        elif action and position == True:
            sell = True
            buy = False
        elif not action and position == True:
            sell = False
            buy = False

        if buy:
            numTrades += 1
            position = True
            shares = amount/price
            balance -= amount
            # print("----- BUY £",amount," -----")
            # print('On date: ',date)
            # print("Bought ",round(shares,4),' shares at price ',price,'\n')

        elif sell and shares > 0:
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
            bhBalance = bhShares*priceData[oldDate]
            break

        oldDate = date
        answer = (((returns+equity)-amount)/amount)*100

    return round(answer,2), numTrades,#, returns+equity



creator.create("Fitness", base.Fitness, weights=(1.0,-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

terminal_types = [so_bounds,so_int,macd_bounds,macd_sig,macd_f,macd_s, ema_int ,ma_int, rsi_int, rsi_bounds, str, position_bool]
toolbox = base.Toolbox()

toolbox.register("expr", gp.generate_safe, pset=pset, min_=1, max_=7, terminal_types=terminal_types)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", simulation)
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.generate_safe, min_=1, max_=5, terminal_types=terminal_types)
# Check mutation is working properly
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)



def showTree(tree):
    nodes, edges, labels = gp.graph(tree)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")

    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()
    return

def plot_pareto_front(paretofront, title=""):
    x=[]
    y=[]
    for p in paretofront:
        print("Ind: "+str(p))
        fitness = p.fitness.values
        print("Fit: "+str(fitness))
        x.append(fitness[0])
        y.append(fitness[1])
    fig,ax=plt.subplots(figsize=(5,5))
    ax.plot(x,y,".")
    ax.set_title(title)
    plt.show()

def plot_pop_pareto_front(pop,paretofront, title=""):
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
    print("Pareto: "+str(fitpareto))
    print("Population: "+str(fitpop))

    ax.set_title(title)
    plt.xlabel('% increase in profit', fontsize=16)
    plt.ylabel('Number of Trades', fontsize=16)
    plt.legend()
    plt.show()

def main(parallel=True):

    random.seed()
    NGEN = 4
    MU = 20
    LAMBDA = 20
    CXPB = 0.5
    MUTPB = 0.2

    # NGEN = 100
    # MU = 100
    # CXPB = 0.8

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
    hof = tools.HallOfFame(1)
    pf = tools.ParetoFront()

    all = []
    # LOOK INTO THIS APPROACH FOR UNPACKING EACH GENERATION
    for x in range(NGEN):
        pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, 1, stats, halloffame=hof)
        for ind in pop:
            all.append(ind)
        if pf is not None:
            pf.update(pop)

    plot_pop_pareto_front(all, pf, "Gen: %d"%(x))

    if parallel:
        pool.close()

    return pop, logbook


if __name__ == "__main__":
    freeze_support()
    random.seed()
    pop, stats = main()
