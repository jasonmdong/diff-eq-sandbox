import matplotlib.pyplot as plt
import math

h = 0.01
t = 5
yn = 1
tn = 1
n = (int) (t / h)

def f(t, y):
    return y

x, y = [], []
for i in range(0, n):
    ynp1 = yn + h * f(tn, yn)
    x.append(tn)
    y.append(yn)
    yn = ynp1
    tn += h

plt.plot(x, y)
plt.savefig("plot.png")