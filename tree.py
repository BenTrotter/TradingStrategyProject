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

def showTree(tree,title):
    plt.close('all')
    nodes, edges, labels = gp.graph(tree)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")

    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    ax = plt.gca()
    ax.set_title(title)
    plt.show()
    plt.close()
    return

if __name__ == "__main__":
    pickle_off = open("SavedOutput.pkl","rb")
    results = pickle.load(pickle_off)

    target = 124.62 # The training score of the desired tree
    count = 0
    for k,v in results.items():
        if count == 2:
            for ind in v:   
                if ind.fitness.values[0] == target:
                    print("FOUND")
                    showTree(ind,ind.fitness.values)
        count += 1
