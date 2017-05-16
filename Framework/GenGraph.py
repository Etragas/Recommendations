from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
import plotly.graph_objs as go
import plotly
import numpy as np
from autograd.util import flatten
import warnings
import pygraphviz as pgv

def numel(items = [], isMovie = 0, names=[[],[]], call_train = {}, data = None):
    print call_train
    if not call_train:
        return None

    for parent in call_train:
        children_dict = call_train[parent]
        for child in children_dict:
            items.append([names[isMovie][parent] , names[not isMovie][child.keys()[0]], str(data[parent,child.keys()[0]]) if not isMovie else str(data[child.keys()[0],parent])])
            numel(items,not isMovie, names, child,data)
    return items

def draw_stuff(names,call_train,data):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        import igraph
        from igraph import Graph,EdgeSeq
        igraph.__version__

        g = pgv.AGraph(directed=True,ranksep='.1',nodesep='.1')
        print call_train

        edges = numel([],0,names,call_train, data)
        print edges
        sublprop = zip(*edges)[:2]
        vertices = set([item for sublist in sublprop for item in sublist])
        #Create mapped edges
        g.add_nodes_from(vertices)
        for edge in edges:
            print "New edge"
            print edge
            g.add_edge(edge[1],edge[0],label=edge[2],dir='back')

        for x in range(100):
            #User
            try:
                n = g.get_node(names[0][x])
                if x < 10:
                    n.attr['shape'] = 'box'
                n.attr['color'] = 'blue'
            except:
                pass
            try:
                n = g.get_node(names[1][x])
                if x < 10 :
                    n.attr['shape'] = 'box'
                n.attr['color'] = 'green'
            except:
                pass
        g = g.reverse()
        g.layout(prog="dot",)
        g.draw('file.png')
        print g
        raw_input()