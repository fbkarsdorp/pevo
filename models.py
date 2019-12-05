import collections

import networkx as nx
import numpy as np
from numpy.random.mtrand import dirichlet
import matplotlib.pyplot as plt
import tqdm

from utils import turnover, turnover_plot, lorenz


class NeutralModel:

    """Neutral Model of Cultural Evolution as described in Bentley et al. (2004).
    In the Neutral model, individuals copy traits from other randomly selected
    individuals. With a small probability mu, they innovate a new trait.

    - Bentley, R. A., Hahn, M. W., & Shennan, S. J. (2004). Random drift and culture change. 
      Proceedings of the Royal Society B: Biological Sciences, 271(1547), 1443-1450. 
      http://doi.org/10.1098/rspb.2004.2746
    """

    def __init__(self, N: int = 100, mu: float = 0.01, T: int = None,
                 yield_population: bool = False, seed: int = None, **kwargs) -> None:
        """
        Initialize the Neutral model.

        Parameters
        ----------
        N : integer, default = 100
            The population size.
        mu : float (0 < mu <= 1), default = 0.01
            The parameter mu controls the probability that 
            individuals copy the behavior from others or innovate.
        T : integer, default = None
            The number of generations to run. When set to None, T an optimal value
            for T will be computed automatically.
        seed : integer, default = None
            Random state seed.
        """
        self.N = N
        self.mu = mu
        self.rnd = np.random.RandomState(seed)
        self.T = int(4 * (mu ** -1) + (mu ** -1) + 50) if T is None else T
        self.yield_population = yield_population
        self.max_traits = int(np.floor(N + N * self.T * mu))
        self.population = np.arange(N)
        self.n_traits = N
        self.freq_traits = np.zeros((self.T, self.max_traits))
        self.parents = np.zeros((self.T, N), dtype=np.int)

    def run(self) -> None:
        "Run the simulation model."
        for t in tqdm.tqdm(range(self.T)):
            if self.n_traits <= self.max_traits:
                parents = self.rnd.randint(self.N, size=self.N)
                models = self.population[parents]
                self.copy(t, models, parents)
                # innovations
                innovations = self.rnd.binomial(1, self.mu, self.N).astype(bool)  # self.rnd.rand(self.N) < self.mu
                n_innovations = innovations.sum()
                self.population[innovations] = np.arange(
                    self.n_traits, self.n_traits + n_innovations)
                self.n_traits = self.n_traits + n_innovations
                # innovators have no parents
                self.parents[t, innovations] = -1
                # compute frequency of traits in time step t
                counts, _ = np.histogram(self.population, np.arange(self.max_traits + 1))
                self.freq_traits[t] = counts
                # if self.yield_population:
                #     yield self.population
        # update idnumbers of parents, each unique for a time step.
        innovations = np.where(self.parents == -1)
        self.parents += np.array([np.arange(self.T) * self.N]).T
        self.parents[innovations] = -1

    def copy(self, t, models, parents) -> None:
        """In the neutral model individuals copy following a
        uniform distribution over the population."""
        self.population = models
        self.parents[t] = parents

    def _data(self) -> np.ndarray:
        span = int(self.mu ** -1)
        time, _ = self.freq_traits.shape
        return self.freq_traits[time - span - 50 - 1: time - 50]

    def turnover_plot(self, span, y=20):
        "Plot a turnover plot as described in Acerbi & Bentley (2014)."
        t = turnover(self._data(), span, y=y).mean(0)
        turnover_plot(t)

    def frequency_distribution(self) -> np.ndarray:
        data = self._data.sum(0)
        return data[data > 0]

    def survival_matrix(self):
        return (self.freq_traits > 0).cumsum(0).argmax(0) - (self.freq_traits > 0).argmax(0) + 1

    def lorenz_curve(self):
        "Plot the Lorenz curve over the parents."
        span = int(self.mu ** -1)
        data = self.parents[self.parents.shape[0] - span - 50 - 1: self.parents.shape[0] - 50]
        data = data.ravel()
        p, c =  lorenz(collections.Counter(data[data > -1]).values())
        plt.plot(c, p, 'o')
        return p, c

    def to_graph(self):
        G = nx.DiGraph()
        span = int(self.mu ** -1)
        data = self.parents[self.parents.shape[0] - span - 50 - 1: self.parents.shape[0] - 50]
        start = self.parents.shape[0] - span - 50 - 1
        for i in range(data.shape[0]):
            parents = data[i]
            children = np.arange(self.N) + self.N * (i + start + 1)
            assert np.all(children > parents)
            for parent, child in zip(parents, children):
                if parent >= 0:
                    G.add_edge(child, parent)
                else:
                    G.add_node(child)
        return G

    def degree_distribution(self):
        span = int(self.mu ** -1)
        data = self.parents[self.parents.shape[0] - span - 50 - 1: self.parents.shape[0] - 50]
        data = data.ravel()
        return collections.Counter(data[data > -1]).values()


class FrequencyIndependentSelectionModel(NeutralModel):
    def __init__(self, N=100, mu=0.01, seed=None, **kwargs):
        super(FrequencyIndependentSelectionModel, self).__init__(N=N, mu=mu, seed=seed, **kwargs)

    def copy(self, t, models, parents):
        # Pick a random cultural trait independently from
        # its frequency distribution in the population
        self.population = self.rnd.randint(self.population.max(), size=self.N)


class ContentBiasedModel(NeutralModel):
    """
    Abstract class representing a content-biased model of Cultural 
    Evolution. In these models, individuals copy traits from other individuals
    on the basis of their knowledge about the frequency or attractiveness of 
    those traits. The implementations roughly follow the descriptions of 
    and Acerbi & Bentley (2014), but differ slightly in the way copying is handled. 
    Whereas in Acerbi & Bentley (2014) individuals may decide not to copy on 
    the basis of their knowledge of the frequency of traits, in this implementation, 
    individuals always copy, either from someone who represents the norm
    or from someone who goes against the norm.

    - Acerbi, A., & Bentley, R. A. (2014). Biases in cultural transmission shape the 
      turnover of popular traits. Evolution and Human Behavior, 35(3), 228-236. 
      http://doi.org/10.1016/j.evolhumbehav.2014.02.003
    """
    def __init__(self, N=100, mu=0.01, C=0.5, seed=None, **kwargs):
        """
        Initialize the ContentBiasedModel

        Parameters
        ----------
        N : integer, default = 100
            The population size.
        mu : float (0 < mu <= 1), default = 0.01
            The parameter mu controls the probability that 
            individuals copy the behavior from others or innovate.
        C : float, default = 0.5
            The parameter C controls the probability that an
            individual is content biased or not. If not, the individual
            we copy at random just as in the Neutral model.
        seed : integer, default = None
            Random state seed.
        """
        super(ContentBiasedModel, self).__init__(N=N, mu=mu, seed=seed, **kwargs)
        self.C = C

    def copy(self, t, models, parents):
        """
        In Content biased models, individuals copy the trait of other
        individuals on the basis of their knowledge of the frequency or 
        attractiveness of that trait. With a probability of C, individuals
        will be biased; the rest (1 - C) copies unbiased.
        """
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

    def select_biased(self, t):
        raise NotImplementedError


class ConformistModel(ContentBiasedModel):
    """
    Class representing the Conformist Model as described in Acerbi & Bentley (2014).
    In this implementation, individuals copy the traits from other individuals if
    they possess a trait that is in the n most frequent traits in the population. 
    Even if an individual already possesses a trait that complies with the norm, 
    (s)he will copy the trait from a randomly selected individual who adheres 
    to the norm in the population.
    """
    def __init__(self, N=100, mu=0.01, n_best=10, C=0.5, seed=None, **kwargs):
        """
        Initialize the ConformistModel.

        Parameters
        ----------
        N : integer, default = 100
            The population size.
        mu : float (0 < mu <= 1), default = 0.01
            The parameter mu controls the probability that 
            individuals copy the behavior from others or innovate.
        n_best : integer, default = 10
            If an individual possesses a trait that is among the n_best
            traits, s(he) is considered to be conform the norm.
        C : float, default = 0.5
            The parameter C controls the probability that an
            individual is content biased or not. If not, the individual
            we copy at random just as in the Neutral model.
        seed : integer, default = None
            Random state seed.
        """
        super(ConformistModel, self).__init__(N=N, mu=mu, C=C, seed=seed, **kwargs)
        self.n_best = n_best

    def select_biased(self, t):
        traits = np.argsort(self.freq_traits[t-1])[::-1]
        return np.all(np.not_equal(np.array([self.population]).T, 
                                   np.kron(np.ones((self.N, self.n_best)), traits[:self.n_best])),
                      axis=1)

class AntiConformistModel(ContentBiasedModel):
    """
    Class representing the Anti-Conformist Model as described in Acerbi & Bentley (2014).
    In this implementation, individuals copy the traits from other individuals if
    they possess a trait that is NOT in the n most frequent traits in the population. 
    Even if an individual already possesses a trait that does NOT comply with the norm, 
    (s)he will copy the trait from a randomly selected individual who goes against the norm.    
    """
    def __init__(self, N=100, mu=0.01, n_best=10, C=0.5, seed=None, **kwargs):
        """
        Initialize the AntiConformistModel.

        Parameters
        ----------
        N : integer, default = 100
            The population size.
        mu : float (0 < mu <= 1), default = 0.01
            The parameter mu controls the probability that 
            individuals copy the behavior from others or innovate.
        n_best : integer, default = 10
            If an individual possesses a trait that is among the n_best
            traits, s(he) is considered to be conform the norm. In the Anti-Conformist
            model, individuals will be selected that possess traits NOT in this top list.
        C : float, default = 0.5
            The parameter C controls the probability that an
            individual is content biased or not. If not, the individual
            we copy at random just as in the Neutral model.
        seed : integer, default = None
            Random state seed.
        """            
        super(AntiConformistModel, self).__init__(N=N, mu=mu, C=C, seed=seed, **kwargs)
        self.n_best = n_best

    def select_biased(self, t):
        traits = np.argsort(self.freq_traits[t-1])[::-1]
        return np.any(np.equal(np.array([self.population]).T, 
                               np.kron(np.ones((self.N, self.n_best)), traits[:self.n_best])), 
                      axis=1)


class AttractionModel(ContentBiasedModel):
    """
    Class representing the Attraction biased model as described in Acerbi & Bentley (2014).
    individuals have knowledge about the inherent attractiveness of certain traits. They
    prefer to copy from individuals who possess traits that are more attractive than their
    own. As in the other content biased models, individuals will always copy from some 
    individual that possesses a more attractive trait.
    """
    def __init__(self, N=100, mu=0.01, C=0.5, seed=None, **kwargs):
        """
        Initialize the AttractionModel.

        Parameters
        ----------
        N : integer, default = 100
            The population size.
        mu : float (0 < mu <= 1), default = 0.01
            The parameter mu controls the probability that 
            individuals copy the behavior from others or innovate.
        C : float, default = 0.5
            The parameter C controls the probability that an
            individual is content biased or not. If not, the individual
            we copy at random just as in the Neutral model.
        seed : integer, default = None
            Random state seed.
        """        
        super(AttractionModel, self).__init__(N=N, mu=mu, seed=seed, **kwargs)
        self.alpha = self.rnd.randn(self.max_traits)

    def select_biased(self, t):
        return self.alpha[self.population] < self.alpha[self.rnd.randint(self.N)]


class ExemplarModel(NeutralModel):
    """
    Class representing the Exemplar-biased model. In this model, individuals
    copy the traits from other individuals in the population, not on the basis
    of the inherent properties of their traits but on the basis of extrinsic 
    properties of those individuals. These extrinsic properties could represent
    prestige or age. The crucial difference with the content-based models is 
    that even though two individuals may possess the same trait, one of them may 
    be preferred over the other because s(he) is more prestigious. 
    """
    def __init__(self, N=100, mu=0.01, C=0.5, alpha=0.1, seed=None, **kwargs):
        """
        Initialize the ExemplarModel

        Parameters
        ----------
        N : integer, default = 100
            The population size.
        mu : float (0 < mu <= 1), default = 0.01
            The parameter mu controls the probability that
            individuals copy the behavior from others or innovate.
        C : float, default = 0.5
            The parameter C controls the probability that an
            individual is content biased or not. If not, the individual
            we copy at random just as in the Neutral model.
        alpha : float (0 < alpha), default = 0.1
            The hyper-parameter alpha of the dirichlet distribution.
        seed : integer, default = None
            Random state seed.
        """
        super(ExemplarModel, self).__init__(N=N, mu=mu, seed=seed, **kwargs)
        self.alpha = alpha
        self.C = C

    def copy(self, t, models, parents):
        """
        In the exemplar based model, individuals copy the trait of other
        individuals on the basis of the extrinsic properties of those individuals.
        With a probability of C, individuals will be biased towards copying from
        such individuals; the rest (1 - C) copies unbiased.
        """
        if t > 0:
            biased_parents = self.rnd.choice(
                self.N, size=self.N, p=dirichlet([self.alpha] * self.N))
            biased = self.rnd.rand(self.N) < self.C
            biased_parents[~biased] = parents[~biased]
            self.population = self.population[biased_parents]
            self.parents[t] = biased_parents
        else:
            self.population = models
            self.parents[t] = parents
