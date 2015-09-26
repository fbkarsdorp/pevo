import numpy as np
import seaborn as sns

from scipy.optimize import curve_fit


def simulate(model, iterations=100):
    turnovers = []
    for iteration in range(iterations):
        m = model.__class__(**model.__dict__)
        m.run()
        span = int(m.mu ** -1)
        data = m.freq_traits[m.freq_traits.shape[0] - span - 50 - 1: m.freq_traits.shape[0] - 50]
        t = turnover(data, span, y=20).mean(0)
        turnovers.append(t)
    return turnovers

def gini_coeff(data):
    d = sorted(data, reverse=True)
    n = len(d)
    sq = 0.0
    for i in range(n):
        if i == 0:
            q0 = d[i]
        else:
            q1 = q0 + d[i]
            q0 = q1
        sq = sq + q0
    s = 2 * sq / sum(d) - 1
    R = n / (n - 1.) * (1. / n * s - 1.)
    return R

def lorenz(data):
    d = sorted(data, reverse=True)
    n, s, p = len(d), sum(d), np.arange(0.0, 1.01, 0.01)
    c = np.zeros(p.shape[0])
    items = np.zeros(p.shape[0])
    i = 0
    for x in p:
        if x == 0:
            items[i] = 0
            c[i] = 0
        else:
            items[i] = int(np.floor(n * x));
            c[i] = sum(d[:int(items[i])]) / float(s)
        i += 1
    return p, c

def turnover(data, span, y=20):
    top_n = np.zeros((span + 1, y))
    turn = np.zeros((span, y))
    k = 0
    for t in range(int(span)):
        trait = np.argsort(data[t])[::-1]
        top_n[k] = trait[:y]
        if t > 0:
            N = np.sum(data[t] > 0)
            if N > y:
                stop = y
            else:
                stop = N
                turn[k-1, N: y] = np.nan
            for tt in range(stop):
                turn[k - 1, tt] = np.setdiff1d(top_n[k][:tt+1], top_n[k - 1][:tt+1]).shape[0]
        k = k + 1
    return turn

def turnover_plot(t):

    def z(x, a, b):
        return a * x ** b

    sns.plt.plot(t, 'o')
    popt, pcov = curve_fit(z, np.arange(t.shape[0]), t)
    print "a = %.3f; b = %.3f" % (popt[0], popt[1])
    sns.plt.plot(z(np.arange(t.shape[0]), *popt), '-k')
    sns.plt.ylabel("z"); sns.plt.xlabel("y")
    sns.plt.show()