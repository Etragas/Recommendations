from autograd import *

j = 3


class dog:
    def f(self, y, a):
        return 2 * a[0] * a[1]


print grad(dog().f, 1)(0, [3.0, 2.0])
