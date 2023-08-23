from math import *

X = [62, 60, 78, 62, 49, 67, 80, 48]
Y = [24, 56, 42, 74, 44, 28]

sumXSquared = 0
sumYSquared = 0

for x in X:
    sumXSquared += x**2

for y in Y:
    sumYSquared += y**2

print("sumsquare X: ", sumXSquared)
print("sumsquare Y: ", sumYSquared)

avgX = sum(X)/len(X)
avgY = sum(Y)/len(Y)
print("avg X: ", avgX)
print("avg Y: ", avgY)

S_X = 0
S_Y = 0

for x in X:
    S_X += (x - avgX)**2

for y in Y:
    S_Y += (y - avgY)**2


Xs = sumXSquared*len(X) - sum(X)**2
Xs /= len(X)*(len(X) - 1)

Ys = sumYSquared*len(Y) - sum(Y)**2
Ys /= len(Y)*(len(Y) - 1)


print("S_x^2: ", Xs)
a1 = (Xs)

print("S_y^2: ", Ys)
a2 = (Ys)

xSS = 0
for x in X:
    xSS += (x-avgX)**2

ySS = 0
for y in Y:
    ySS += (y-avgY)**2

SP = sqrt((xSS + ySS)/(len(X) + len(Y) - 2))

print("SP: ", SP)

t = (avgX - avgY)/(SP*sqrt(1/len(X) + 1/len(Y)))

print("t: ", t)

# X = [8, 4, 6, 3, 1, 4, 4, 6, 4, 2, 2, 1, 1, 4, 3, 3, 2, 6, 3, 4]
# X = [2, 1, 1, 3, 2, 7, 2, 1, 3, 1, 0, 2, 4, 2, 3, 3, 0, 1, 2, 2]

# sumXSquared = 0

# for x in X:
#     sumXSquared += x**2

# print(sumXSquared)

# avg = sum(X)/len(X)
# print(avg)
# S = 0

# for x in X:
#     S += (x - avg)**2

# print(S)

# Xs = sumXSquared*20 - sum(X)**2
# Xs /= 20*(20-1)

# print(Xs)
# print(sqrt(Xs))

# b1 = (Xs)

# print(b1/a1)
