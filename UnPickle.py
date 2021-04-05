import random
import operator
import csv
import itertools
from preload import load

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import numpy

import deap
from importlib import reload
reload(deap)

from HPCEvolution import *
from preload import load

from deap import algorithms, base, creator, tools, gp
from deap.benchmarks.tools import diversity, convergence, hypervolume

from datetime import datetime, timedelta
import pandas as pd

import pickle

from scoop import futures

# ----------------- Parameters -------------------- #
unseenStart = "2015-01-01"
unseenEnd = "2021-01-01"
interval = 6
file = "MSFT.csv"
# ------------------------------------------------- #

if __name__ == "__main__":
    pickle_off = open("SavedOutput.pkl","rb")
    results = pickle.load(pickle_off)

    load("MSFT",unseenStart,unseenEnd,'1d')

    count = 0
    for k, v in results.items():
        if count == 2:
            answer, interval1 = unseen(v, unseenStart, unseenEnd, interval, file)
            processPareto(answer,interval1)

        count += 1




        
