from collections import defaultdict, namedtuple

import numpy as np
from scipy.sparse import dok_matrix
from sortedcontainers import SortedList

MatrixIndices = namedtuple('MatrixIndices', ['rows', 'cols'])


class non_zero_hero(dok_matrix):

    def freeze_dataset(self):
        print("Num items", len(dict.items(self)))
        self.non_zero_for_row = defaultdict(lambda: SortedList())
        self.non_zero_for_col = defaultdict(lambda: SortedList())

        self.__setitem__ = self.finalizeSet
        for key in dict.keys(self):
            row, col = key
            self.non_zero_for_row[row].add(val=col)
            self.non_zero_for_col[col].add(val=row)
        self.nonzero_indices = list(zip(*self.nonzero()))
        self.num_nonzero = len(self.nonzero_indices)

    def get_random_indices(self, numIndices):
        choices = np.random.randint(0, self.num_nonzero, numIndices)
        # choices = np.random.choice(self.num_nonzero, numIndices, replace=False)
        return (self.nonzero_indices[choice] for choice in choices)

    def get_non_zero(self, row=None, col=None) -> MatrixIndices:

        if not row and not col:
            return None

        if not col:
            col = self.non_zero_for_row[row]

        if not row:
            row = self.non_zero_for_col[col]

        return MatrixIndices(rows=row, cols=col)

    def finalizeSet(self, index, x):
        raise AssertionError("You cannot insert a value after finalization")
