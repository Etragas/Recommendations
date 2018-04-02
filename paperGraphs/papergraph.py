import pygraphviz as pgv
G = pgv.AGraph(directed=True)
nodelist = ['u1','u2','u3','v1','v2','v3','ui','vj','?']
G.add_nodes_from(nodelist,style='filled')
def add_edge(*args,**args1):
    return G.add_edge(*args,dir='back',**args1)
add_edge('ui','vj',color='blue')
add_edge('u1','v3',color='red')
add_edge('u3','vj',color='blue')
G.get_node('u3').attr['style'] = 'empty'
G.get_node('v3').attr['style'] = 'empty'
G.get_node('ui').attr['style'] = 'empty'
G.get_node('vj').attr['style'] = 'empty'
G.get_node('?').attr['style'] = 'empty'
add_edge('u2','v3',color='red')
add_edge('u2','vj',color='blue')
add_edge('v1','u3',color='cyan')
add_edge('v1','u4',color='cyan')
add_edge('v2','u3',color='cyan')
add_edge('v3','u3',color='cyan')
add_edge('v3','ui',color='cyan')
add_edge('v1','ui',color='cyan')
add_edge('vj','?',color='green')
add_edge('ui','?',color='green')
G = G.reverse()
G.layout(prog='dot') # use dot
G.draw('file.png')  # write previously positioned graph to PNG file
print(G)

