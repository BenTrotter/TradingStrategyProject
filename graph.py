import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy


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
    ax.plot(xp,yp,".", label="All Solutions")
    ax.plot(x,y,".", label="Pareto Front")
    fitpareto=list(zip(x,y))
    fitpop=list(zip(xp,yp))


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
    if option == 7:
        plt.xlabel('% increase in profit')
        plt.ylabel('Risk Exposure')
    if option == 8:
        plt.xlabel('% increase in profit')
        plt.ylabel('Number of trades')
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
    elif option == 9:
        df = pd.DataFrame(allValues, columns=['Profit','Risk Exposure','No. Trades'])
    elif option == 10:
        df = pd.DataFrame(allValues, columns=['Profit','Sharpe Ratio','Risk Exposure','No. Trades'])
    elif option == 11:
        df = pd.DataFrame(allValues, columns=['PC','Sharpe Ratio','Risk Exposure','No. Trades'])

    # colors = iris['species'].replace({'setosa':'red', 'virginica': 'green', 'versicolor':'blue'})   
    # pd.plotting.scatter_matrix(iris, c=colors)
    scatter = pd.plotting.scatter_matrix(df, alpha=0.2,hist_kwds={'color':'red'})
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
    elif option == 9:
        ax.set_xlabel('% increase in profit', rotation=150)
        ax.set_ylabel('Risk Exposure')
        ax.set_zlabel('No. Trades', rotation=60)
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


def scatter(all,pareto,option):

    allValues = []
    for i in all:
        allValues.append(i.fitness.values)

    allPareto = []
    for x in pareto:
        allPareto.append(x.fitness.values)

    d = {}
    target = []
    for ind in allValues:
        if ind in allPareto:
            target.append(1)
        else:
            target.append(0)
    
    d['data'] = allValues
    d['target'] = target

    if option == 2:
        d['target_names'] = ['Profit','PC','Risk Exposure']
    elif option == 3:
        d['target_names'] = ['Profit','PC','Risk Exposure','No. Trades']
    elif option == 6:
        d['target_names'] = ['Profit','PC','Sharpe Ratio']
    elif option == 9:
        d['target_names'] = ['Profit','Risk Exposure','No. Trades']
    elif option == 10:
        d['target_names'] = ['Profit','Sharpe Ratio','Risk Exposure','No. Trades']
    elif option == 11:
        d['target_names'] = ['PC','Sharpe Ratio','Risk Exposure','No. Trades']


    X = d['data']
    y = d['target']
    df = pd.DataFrame(X, columns = d['target_names'])

    pd.plotting.scatter_matrix(df, c=y, figsize = [8,8],
                        s=80, marker = 'D');
    df['y'] = y
    sns.set(style="ticks", color_codes=True)

    sns.pairplot(df,hue='y')

    plt.rcParams["figure.subplot.right"] = 0.8


    handles = [plt.plot([],[], ls="", marker=".", \
                    markersize=numpy.sqrt(200))[0] for i in range(2)]
    labels=["All solutions", "Pareto Front"]
    plt.legend(handles, labels)
 
    plt.savefig(r"scatter.png")