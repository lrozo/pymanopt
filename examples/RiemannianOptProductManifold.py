import autograd.numpy as np
from autograd.scipy.special import logsumexp
from pymanopt.manifolds import Product, Euclidean, SymmetricPositiveDefinite
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions, ConjugateGradient
import matplotlib.pyplot as plt
import pymanopt

# Number of data
N = 1000
# Dimension of data
D = 2
# Number of clusters
K = 3
# GMM parameters to generate samples
pi = [0.1, 0.6, 0.3]
mu = [np.array([-4, 1]), np.array([0, 0]), np.array([2, -1])]
Sigma = [np.array([[3, 0],[0, 1]]), np.array([[1, 1.], [1, 3]]), .5 * np.eye(2)]
components = np.random.choice(K, size=N, p=pi)
samples = np.zeros((N, D))
# for each component, generate all needed samples
for k in range(K):
    # indices of current component in X
    indices = (k == components)
    # number of those occurrences
    n_k = indices.sum()
    if n_k > 0:
        samples[indices] = np.random.multivariate_normal(mu[k], Sigma[k], n_k)

colors = ['r', 'g', 'b', 'c', 'm']
for i in range(K):
    indices = (i == components)
    plt.scatter(samples[indices, 0], samples[indices, 1], alpha=.4, color=colors[i%K])
plt.axis('equal')
plt.show()

# (1) Instantiate the manifold
manifold = Product([SymmetricPositiveDefinite(D+1, k=K), Euclidean(K-1)])

# (2) Define cost function
# The parameters must be contained in a list theta.
@pymanopt.function.Autograd
def cost(S, v):
    # Unpack parameters
    nu = np.append(v, 0)

    logdetS = np.expand_dims(np.linalg.slogdet(S)[1], 1)
    y = np.concatenate([samples.T, np.ones((1, N))], axis=0)

    # Calculate log_q
    y = np.expand_dims(y, 0)

    # 'Probability' of y belonging to each cluster
    log_q = -0.5 * (np.sum(y * np.linalg.solve(S, y), axis=1) + logdetS)

    alpha = np.exp(nu)
    alpha = alpha / np.sum(alpha)
    alpha = np.expand_dims(alpha, 1)

    loglikvec = logsumexp(np.log(alpha) + log_q, axis=0)
    return -np.sum(loglikvec)


problem = Problem(manifold=manifold, cost=cost, verbosity=2)

# (3) Instantiate a Pymanopt solver
solver = SteepestDescent()

# let Pymanopt do the rest
Xopt = solver.solve(problem)
print("Done!")

