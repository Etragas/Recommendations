from collections import namedtuple
import numpy as np
import pandas

import pstats
import marshal

CallStats = namedtuple('CallStats', ['n_calls', 'n_non_rec_calls', 'total_time', 'cumulative_time', 'subcall_statistics'])

def getRatio(cs1, cs2):
    pairs = list(zip(cs1,cs2))
    return CallStats(*map(lambda tup: tup[0]/tup[1] if tup[0] >0 and tup[1] >0 else 0, pairs[:-1]),0)

def createDict(path):
    statDict= {}
    with open(path, 'rb') as ml100k:
        stats = marshal.load(ml100k)
        for item in stats.items():
            name = "".join([str(x) for x in item[0]])
            callStats = CallStats(*item[1])
            # print("Name ", name)
            # print("Stats ", callStats)
            statDict[name] = callStats
    return statDict

# p = pstats.Stats('../ml-100k-stats')
ml100kMap = createDict('../ml-100k-stats')
# ml1mMap = createDict('../ml-1m-stats')

ratios = {}
for key in ml100kMap.keys():
    if key not in ml100kMap:
        print("Key {} not in ml100k".format(key))
        continue
    # print(ml1mMap[key])
    # print(ml100kMap[key])
    ratios[key] = ml100kMap[key]

ratioList = [(k,v) for k,v in ratios.items()]
ratioList = sorted(ratioList,key=lambda t: t[1].cumulative_time / t[1].n_calls, reverse=True)
for x in ratioList:
    print(x)
# l = p.strip_dirs().sort_stats('ncalls')
# for stat in l.sort_arg_dict:
#     print(stat)
