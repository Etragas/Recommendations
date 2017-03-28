import numpy as np

def nested_sum(l1,l2):
    #Do rec over one, adding from other
    for idx in range(len(l1)):
        if type(l1) is np.ndarray:
            l1[idx] = l1[idx] + l2[idx]
        else:
            l1[idx] = nested_sum(l1[idx],l2[idx])
    return l1

l1 = [[np.array([1,2,3]),np.array([4,5])],np.array([6])]
l2 = [[np.array([6,5,4]),np.array([3,2])],np.array([1])]

print(nested_sum(l1,l2))
