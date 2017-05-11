from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
import plotly.graph_objs as go
import plotly
import numpy as np
from autograd.util import flatten
import warnings
import pygraphviz as pgv

def numel(items = [], isMovie = 0, names=[[],[]], call_train = {}):
    print call_train
    if not call_train:
        return None

    for parent in call_train:
        children_dict = call_train[parent]
        for child in children_dict:
            items.append([names[isMovie][parent] , names[not isMovie][child.keys()[0]]])
            numel(items,not isMovie, names, child)
    return items

def draw_stuff(names,call_train):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        import igraph
        from igraph import Graph,EdgeSeq
        igraph.__version__

        g = pgv.AGraph(directed=True)
        print call_train

        edges = numel([],0,names,call_train)
        print edges
        vertices = set([item for sublist in edges for item in sublist])
        #Create mapped edges
        g.add_nodes_from(vertices)
        g.add_edges_from(edges)
        for x in range(10):
            #User
            try:
                n = g.get_node(names[0][x])
                n.attr['shape'] = 'box'
            except:
                pass
            try:
                n = g.get_node(names[1][x])
                n.attr['shape'] = 'diamond'
            except:
                pass
        g = g.reverse()
        g.layout(prog="dot")
        g.draw('file.png')
        print g
        raw_input()