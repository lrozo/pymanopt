import os
import argparse
import numpy as np
import torch
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import TrustRegions

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def _parse_arguments(name):
    parser = argparse.ArgumentParser(name)
    parser.add_argument("-b", "--backend", help="backend to run the test on", default="PyTorch")
    parser.add_argument("-q", "--quiet", action="store_true")
    return vars(parser.parse_args())


class ExampleRunner:
    def __init__(self, run_function, name):
        self._arguments = _parse_arguments(name)
        self._run_function = run_function
        self._name = name

    def run(self):
        backend = self._arguments["backend"]
        quiet = self._arguments["quiet"]
        if not quiet:
            print(self._name)
            print("-" * len(self._name))
            print("Using '{:s}' backend".format(backend))
            print()
        self._run_function(quiet=quiet)



def run(quiet=True):
    dimension = 3
    num_samples = 200
    num_components = 2
    samples = np.random.randn(num_samples, dimension) @ np.diag([3, 2, 1])
    samples -= samples.mean(axis=0)
    samples_ = torch.from_numpy(samples)

    @pymanopt.function.PyTorch
    def cost(w):
        projector = torch.matmul(w, torch.transpose(w, 1, 0))
        return torch.norm(samples_ - torch.matmul(samples_, projector)) ** 2

    manifold = Stiefel(dimension, num_components)
    problem = pymanopt.Problem(manifold, cost, egrad=None, ehess=None)
    if quiet:
        problem.verbosity = 0

    solver = TrustRegions()
    # from pymanopt.solvers import ConjugateGradient
    # solver = ConjugateGradient()
    estimated_span_matrix = solver.solve(problem)

    if quiet:
        return

    estimated_projector = estimated_span_matrix @ estimated_span_matrix.T

    eigenvalues, eigenvectors = np.linalg.eig(samples.T @ samples)
    indices = np.argsort(eigenvalues)[::-1][:num_components]
    span_matrix = eigenvectors[:, indices]
    projector = span_matrix @ span_matrix.T

    print("Frobenius norm error between estimated and closed-form projection "
          "matrix:", np.linalg.norm(projector - estimated_projector))


if __name__ == "__main__":
    runner = ExampleRunner(run, "PCA")
    runner.run()