import cntk as C
from ipdb import set_trace

class SimpleNet(object):
    def __init__(self):
        self.x = C.input_variable(shape=(1,))
        self.y = C.layers.Dense(1, activation=None, init=C.uniform(1), bias=False)(x)
        self.t = C.input_variable(shape=(1,))
        self.loss = C.reduce_l2(self.y - self.t)

set_trace()

net = SimpleNet()
