from collections import Counter

import numpy as np
from numpy.random.mtrand import dirichlet
import seaborn as sns
import pyprind

from utils import turnover, turnover_plot, lorenz


class NeutralModel(object):
    """Neutral Model of Cultural Evolution as by 
    Bentley et al. (2004) and  Acerbi & Bentley (2014).
    
    - Bentley, R. A., Hahn, M. W., & Shennan, S. J. (2004). Random drift and culture change. 
      Proceedings of the Royal Society B: Biological Sciences, 271(1547), 1443-1450. 
      http://doi.org/10.1098/rspb.2004.2746
    - Acerbi, A., & Bentley, R. A. (2014). Biases in cultural transmission shape the 
      turnover of popular traits. Evolution and Human Behavior, 35(3), 228-236. 
      http://doi.org/10.1016/j.evolhumbehav.2014.02.003
    """

    def __init__(self, N=100, mu=0.01, seed=None, **kwargs):
        """Initialize the Neutral model.
            
        Parameters
        ----------
        N : integer, default = 100
            The population size.
        mu : float (0 < mu <= 1), default = 0.01
            The parameter mu controls the probability that 
            individuals copy the behavior from others or innovate.
        seed : integer, default = None
            Random state seed.
        """
        self.N = N
        self.mu = mu
        self.rnd = np.random.RandomState(seed)
        self.T = int(4 * (mu ** -1) + (mu ** -1) + 50)
        self.max_traits = int(np.floor(N + N * self.T * mu))
        self.population = np.arange(N)
        self.n_traits = N
        self.freq_traits = np.zeros((self.T, self.max_traits))
        self.parents = np.zeros((self.T, N), dtype=np.int)

    def run(self):
        "Run the simulation model."
        self.progress = pyprind.ProgBar(self.T)        
        for t in range(self.T):
            if self.n_traits <= self.max_traits:
                parents = self.rnd.randint(self.N, size=self.N)
                models = self.population[parents]
                self.copy(t, models, parents)
                # innovations
                innovations = self.rnd.rand(self.N) < self.mu
                n_innovations = innovations.sum()
                self.population[innovations] = np.arange(
                    self.n_traits, self.n_traits + n_innovations)
                self.n_traits = self.n_traits + n_innovations
                # innovators have no parents
                self.parents[t, innovations] = -1
                # compute frequency of traits in time step t
                counts, _ = np.histogram(self.population, np.arange(self.max_traits + 1))
                self.freq_traits[t] = counts
                self.progress.update()
        # update idnumbers of parents, each unique for a time step.
        innovations = np.where(self.parents == -1)
        self.parents += np.array([np.arange(self.T) * self.N]).T
        self.parents[innovations] = -1

    def copy(self, t, models, parents):
        """In the neutral model individuals copy following a 
        uniform distribution over the population."""
        self.population = models
        self.parents[t] = parents

    def turnover_plot(self, y=20):
        "Plot a turnover plot as described in Acerbi & Bentley (2014)."
        span = int(self.mu ** -1)
        data = self.freq_traits[self.freq_traits.shape[0] - span - 50 - 1: self.freq_traits.shape[0] - 50]
        t = turnover(data, span, y=20).mean(0)
        turnover_plot(t)

    def lorenz_curve(self):
        "Plot the Lorenz curve over the parents."
        span = int(self.mu ** -1)
        data = self.parents[self.parents.shape[0] - span - 50 - 1: self.parents.shape[0] - 50]
        data = data.ravel()
        p, c =  lorenz(Counter(data[data > -1]).values())
        sns.plt.plot(p, c, 'o')


class ContentBiasedModel(NeutralModel):
    def __init__(self, N=100, mu=0.01, C=0.5, seed=None):
        super(ContentBiasedModel, self).__init__(N=N, mu=mu, seed=seed)
        self.C = C

    def copy(self, t, models, parents):
        if t > 0:
            copy_biased = self.select_biased(t)
            biased = self.rnd.rand(self.N) < self.C
            if biased.sum() > 0 and (~copy_biased).sum() > 0:
                biased_parents = self.rnd.choice(np.nonzero(~copy_biased)[0], size=biased.sum())
                parents[biased] = biased_parents
            self.population = self.population[parents]
            self.parents[t] = parents
        else:
            self.population = models
            self.parents[t] = parents


class ConformistModel(ContentBiasedModel):
    def __init__(self, N=100, mu=0.01, n_best=10, C=0.5, seed=None, **kwargs):
        super(ConformistModel, self).__init__(N=N, mu=mu, C=C, seed=seed)
        self.n_best = n_best

    def select_biased(self, t):
        traits = np.argsort(self.freq_traits[t-1])[::-1]
        return np.all(np.not_equal(np.array([self.population]).T, 
                                   np.kron(np.ones((self.N, self.n_best)), traits[:self.n_best])),
                      axis=1)


class AntiConformistModel(ContentBiasedModel):
    def __init__(self, N=100, mu=0.01, n_best=10, C=0.5, seed=None, **kwargs):
        super(AntiConformistModel, self).__init__(N=N, mu=mu, C=C, seed=seed)
        self.n_best = n_best

    def select_biased(self, t):
        traits = np.argsort(self.freq_traits[t-1])[::-1]
        return np.any(np.equal(np.array([self.population]).T, 
                               np.kron(np.ones((self.N, self.n_best)), traits[:self.n_best])), 
                      axis=1)


class AttractionModel(ContentBiasedModel):
    def __init__(self, N=100, mu=0.01, C=0.5, seed=None, **kwargs):
        super(AttractionModel, self).__init__(N=N, mu=mu, seed=seed)
        self.alpha = self.rnd.randn(self.max_traits)

    def select_biased(self, t):
        return self.alpha[self.population] < self.alpha[self.rnd.randint(self.N)]



class ExemplarModel(NeutralModel):
    def __init__(self, N=100, mu=0.01, C=0.5, alpha=0.1, seed=None, **kwargs):
        super(ExemplarModel, self).__init__(N=N, mu=mu, seed=seed)
        self.alpha = alpha
        self.C = C

    def copy(self, t, models, parents):
        if t > 0:
            biased_parents = self.rnd.choice(self.N, size=self.N, p=dirichlet([self.alpha] * self.N))
            biased = self.rnd.rand(self.N) < self.C
            biased_parents[~biased] = parents[~biased]
            self.population = self.population[biased_parents]
            self.parents[t] = biased_parents
        else:
            self.population = models
            self.parents[t] = parents

