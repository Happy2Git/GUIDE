"""The pyramid match kernel as in :cite:`nikolentzos2017matching`."""
# revised from https://github.com/ysig/GraKeL/blob/master/grakel/kernels/pyramid_match.py 
# Author: Cheng-Long Wang <chenglong.wang@kaust.edu.sa>
# License: BSD 3 clause

import collections
import warnings

import numpy as np
from torch_geometric.utils import to_dense_adj

from itertools import chain
from six import itervalues
from six import iteritems

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


class PyramidMatchVector():
    """Pyramid match PyramidMatch class.
    Kernel defined in :cite:`nikolentzos2017matching`
    Parameters
    ----------
    with_labels : bool, default=True
        A flag that determines if the kernel computation will consider labels.
    L : int, default=4
        Pyramid histogram level.
    d : int, default=6
        The dimension of the hypercube.
    Attributes
    ----------
    _num_labels : int
        The number of distinct labels, on the fit data.
    _labels : dict
        A dictionary of label enumeration, made from fitted data.
    """

    def __init__(self, n_jobs=None,
                 normalize=True,
                 with_labels=False,
                 L=4,
                 d=6):
        """Initialise a `pyramid_match` kernel."""
        self._n_jobs = n_jobs
        self._normalize = normalize
        self.with_labels = with_labels
        self.L = L
        self.d = d
        self._parallel = None

    def kernel_similarity(self, X, Y):
        """Fit and transform, on the two dataset.
        Parameters
        ----------
        X : iterable
            One element

        Y : iterable
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that fitting the given graph
            format). If None the kernel matrix is calculated upon fit data.
            The test samples.

        Returns
        -------
        K : numpy array, shape = [n_targets, n_input_graphs]
            corresponding to the kernel vector, a calculation between
            all pairs between target and input graphs
        """
        self._method_calling = 2
        self._is_transformed = False
        self._method_calling = 1
        self.X = X
        self.Y = Y

        # Transform - calculate kernel matrix
        km = self._calculate_kernel_vector()
        if self._normalize:
            return km[:, 1:] / km[:, :1]
        else:
            return km

    def parse_input(self, X):
        """Parse and create features for pyramid_match kernel.
        Parameters
        ----------
        X : iterable
            For the input to pass the test, we must have:
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that correspond to the given
            graph format). A valid input also consists of graph type objects.
        Returns
        -------
        H : list
            A list of lists of Histograms for all levels for each graph.
        """
        if not isinstance(X, collections.abc.Iterable):
            raise TypeError('input must be an iterable\n')
        else:
            i = 0
            Us = []
            if self.with_labels:
                Ls = []
            for (idx, x) in enumerate(X):
                A = to_dense_adj(x).squeeze().numpy()
                i += 1
                if A.shape[0] == 0:
                    Us.append(np.zeros((1, self.d)))
                else:
                    # Perform eigenvalue decomposition.
                    # Rows of matrix U correspond to vertex representations
                    # Embed vertices into the d-dimensional space
                    if A.shape[0] > self.d+1:
                        # If size of graph smaller than d, pad with zeros
                        # Lambda, U = eigs(csr_matrix(A, dtype=np.float),
                        #                  k=self.d, ncv=10*self.d)
                        U, _, _ = svds(csr_matrix(
                            A, dtype=np.float32), k=self.d)
                        # idx = Lambda.argsort()[::-1]
                        # U = U[:, idx]
                    else:
                        Lambda, U = np.linalg.eig(A)
                        idx = Lambda.argsort()[::-1]
                        U = U[:, idx]
                        U = U[:, :self.d]
                    # Replace all components by their absolute values
                    U = np.absolute(U)
                    Us.append((A.shape[0], U))

        if i == 0:
            raise ValueError('parsed input is empty')

        if self.with_labels:
            # Map labels to values between 0 and |L|-1
            # where |L| is the number of distinct labels
            if self._method_calling in [1, 2]:
                self._num_labels = 0
                self._labels = set()
                for L in Ls:
                    self._labels |= set(itervalues(L))
                self._num_labels = len(self._labels)
                self._labels = {l: i for (i, l) in enumerate(self._labels)}
                return self._histogram_calculation(Us, Ls, self._labels)

            elif self._method_calling == 3:
                labels = set()
                for L in Ls:
                    labels |= set(itervalues(L))
                rest_labels = labels - set(self._labels.keys())
                nouveau_labels = dict(chain(iteritems(self._labels),
                                      ((j, i) for (i, j) in enumerate(rest_labels, len(self._labels)))))
                return self._histogram_calculation(Us, Ls, nouveau_labels)
        else:
            return self._histogram_calculation(Us)

    def _histogram_calculation(self, Us, *args):
        """Calculate histograms.
        Parameters
        ----------
        Us : list
            List of tuples with the first element corresponding to the
            number of vertices of a graph and the second to it's
            corresponding to vertex embeddings on the d-dimensional space.
        Ls : list, optional
            List of labels corresponding to each graph.
            If provided the histograms are calculated with labels.
        Labels : dict, optional
            A big dictionary with enumeration of labels.
        Returns
        -------
        Hs : list
            List of histograms for each graph.
        """
        Hs = list()
        if len(args) == 0:
            for (i, (n, u)) in enumerate(Us):
                du = list()
                if n > 0:
                    for j in range(self.L):
                        # Number of cells along each dimension at level j
                        k = 2**j
                        # Determines the cells in which each vertex lies
                        # along each dimension since nodes lie in the unit
                        # hypercube in R^d
                        D = np.zeros((self.d, k))
                        T = np.floor(u*k)
                        T[np.where(T == k)] = k-1
                        for p in range(u.shape[0]):
                            if p >= n:
                                break
                            for q in range(u.shape[1]):
                                # Identify the cell into which the i-th
                                # vertex lies and increase its value by 1
                                D[q, int(T[p, q])] += 1
                        du.append(D)
                Hs.append(du)

        elif len(args) > 0:
            Ls = args[0]
            Labels = args[1]
            num_labels = len(Labels)
            for (i, ((n, u), L)) in enumerate(zip(Us, Ls)):
                du = list()
                if n > 0:
                    for j in range(self.L):
                        # Number of cells along each dimension at level j
                        k = 2**j
                        # To store the number of vertices that are assigned
                        # a specific label and lie in each of the 2^j cells
                        # of each dimension at level j
                        D = np.zeros((self.d*num_labels, k))
                        T = np.floor(u*k)
                        T[np.where(T == k)] = k-1
                        for p in range(u.shape[0]):
                            if p >= n:
                                break
                            for q in range(u.shape[1]):
                                # Identify the cell into which the i-th
                                # vertex lies and increase its value by 1
                                D[Labels[L[p]]*self.d + q, int(T[p, q])] += 1
                        du.append(D)
                Hs.append(du)
        return Hs

    def pairwise_operation(self, x, y):
        """Calculate a pairwise kernel between two elements.
        Parameters
        ----------
        x, y : dict
            Histograms as produced by `parse_input`.
        Returns
        -------
        kernel : number
            The kernel value.
        """
        k = 0
        if len(x) != 0 and len(y) != 0:
            intersec = np.zeros(self.L)
            for (p, xp, yp) in zip(range(self.L), x, y):
                # Calculate histogram intersection
                # (eq. 6 in :cite:`nikolentzos2017matching`)
                if xp.shape[0] < yp.shape[0]:
                    xpp, ypp = xp, yp[:xp.shape[0], :]
                elif yp.shape[0] < xp.shape[0]:
                    xpp, ypp = xp[:yp.shape[0], :], yp
                else:
                    xpp, ypp = xp, yp
                intersec[p] = np.sum(np.minimum(xpp, ypp))
                k += intersec[self.L-1]
                for p in range(self.L-1):
                    # Computes the new matches that occur at level p.
                    # These matches weight less than those that occur at
                    # higher levels (e.g. p+1 level)
                    k += (1.0/(2**(self.L-p-1)))*(intersec[p]-intersec[p+1])
        return k

    def _calculate_kernel_vector(self):
        """Calculate the kernel matrix given a target_graph and a kernel.
        Each a matrix is calculated between all elements of Y on the rows and
        all elements of X on the columns.
        Parameters
        ----------
        Y : list, default=None
            A list of graph type objects. If None kernel is calculated between
            X and itself.
        Returns
        -------
        K : numpy array, shape = [n_targets, n_inputs]
            The kernel matrix: a calculation between all pairs of graphs
            between targets and inputs. If Y is None targets and inputs
            are the taken from self.X. Otherwise Y corresponds to targets
            and self.X to inputs.
        """
        K = np.zeros(shape=(len(self.Y)+1, len(self.X)))
        for (i, x) in enumerate(self.X):
            K[0, i] = self.pairwise_operation(x, x)
        for (j, y) in enumerate(self.Y):
            for (i, x) in enumerate(self.X):
                K[j+1, i] = self.pairwise_operation(y, x)
        return K.T


def indexes(n_jobs, nsamples):
    """Distribute samples accross n_jobs."""
    n_jobs = n_jobs

    if n_jobs >= nsamples:
        for i in range(nsamples):
            yield (i, i+1)
    else:
        ns = nsamples/n_jobs
        start = 0
        for i in range(n_jobs-1):
            end = start + ns
            yield (int(start), int(end))
            start = end
        yield (int(start), nsamples)


def assign(data, K, pairwise_operation):
    """Assign list values of an iterable to a numpy array while calculating a pairwise operation."""
    for d in data:
        K[d[0][0], d[0][1]] = pairwise_operation(d[1][0], d[1][1])
