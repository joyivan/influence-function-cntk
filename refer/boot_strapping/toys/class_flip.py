import numpy as np
from ipdb import set_trace

def class_flip(cls, num_classes, p):
    # cls: 1 dimension integer class
    # num_classes: number of classes
    # p: probability p (0 <= p <= 1)
    rand = np.random.rand(1)[0]
    if rand < p:
        # with probability p, class information is conserved
        return cls
    else:
        # with probability 1-p, class is flipped uniformly
        idx = np.random.randint(num_classes-1)
        clss = list(range(num_classes))
        clss.remove(cls)
        return clss[idx]

#set_trace()
y_hat = []

y = np.ones(1000, np.int)
num_classes = 3
for lb in y:
    y_hat.append(class_flip(lb, num_classes, 0))

set_trace()
print(np.mean(y_hat))
#output = list(map(lambda x: class_flip(x, num_classes, 0.5), y))

print('EOC')

