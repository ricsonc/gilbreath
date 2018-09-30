#!/usr/bin/env python2

import matplotlib.pyplot as plt
import numpy as np
import math

with open("log", "r") as f:
    lines = f.readlines()

nums = []
for line in lines:
    num = int(line.split(' ')[1])
    nums.append(num)

ewma = []
decay = 0.99
current = 400.0
for i, num in enumerate(nums):
    if i > 100:
        decay = 1.0 - 0.1/math.sqrt(i)
    current *= decay
    current += (1-decay)*num
    ewma.append(current)

maxs = []
curmax = 0
for num in nums:
    curmax = max(curmax, num)
    maxs.append(curmax)

#xs = map(lambda x: x*10**10,range(len(ewma)))
xs = range(len(ewma))

plt.semilogx(xs, ewma)
plt.semilogx(xs, maxs)
plt.show()

plt.hist(nums, bins = 16, log = True)
plt.show()
