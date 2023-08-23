from math import *

X = [35, 37, 33, 34, 38, 40, 35, 36, 38, 33, 28, 34, 47, 42, 46]

avg = sum(X)/len(X)

cur = 0

for x in X:
    cur += (x - avg)**2

cur = cur / (len(X) - 1)

print(sqrt(cur))
