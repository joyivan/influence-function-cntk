import time
from ipdb import set_trace

def fooyd(n):
  for i in range(n):
    yield list(range(i))

def foolh(n):
  return [range(i) for i in range(n)]

def full(n):
  a = []
  for i in range(n):
    a.append(list(range(i)))
  return a

def check(n):
  t1 = time.time()

  sm = 0
  for i in full(n): 
    for j in i: 
      sm += 1
  t2 = time.time()
  print('answer is', sm, 'full takes', t2-t1)
  
  sm = 0
  for i in foolh(n): 
    for j in i: 
      sm += 1
  t3 = time.time()
  print('answer is', sm, 'foolh takes', t3-t2)
  
  sm = 0
  for i in fooyd(n): 
    for j in i: 
      sm += 1
  t4 = time.time()
  print('answer is', sm, 'fooyd takes', t4-t3)

def check_full(n):
  t1 = time.time()

  sm = 0
  for i in full(n): 
    for j in i: 
      sm += 1
  t2 = time.time()
  print('answer is', sm, 'full takes', t2-t1)

def check_fooyd(n):
  t1 = time.time()

  sm = 0
  for i in fooyd(n): 
    for j in i: 
      sm += 1
  t2 = time.time()
  print('answer is', sm, 'fooyd takes', t2-t1)

def check_f(n,f):
  sm = 0
  for i in f(n):
    for j in i:
      sm += 1
  print('answer is', sm)


#check_full(1e5) or check_fooyd(1e5)
set_trace()

