from itertools import combinations
from multiprocessing import Pool

import numpy as np
from scipy.spatial.distance import pdist, squareform

from ComplejosSimpliciales.src.SimplicialComplex import SimplicialComplex


class Vietoris_RipsComplex(SimplicialComplex):
    """
    Class used to represent a Vietoris_RipsComplex

    Attributes

    Attributes:

    points (np.array): stores the complex points
    combinations (set): all faces combinations
    """

    def __init__(self, points: np.array, max_size: int = None) -> None:
        """
        Args:
            points (np.array): array of points

        Returns:
            None: instantiates new Vietoris_RipsComplex
        """
        self.points = points
        self.distances = squareform(pdist(points))
        self.max_size = max_size

        # Compute the Vietoris-Rips complex
        self.faces, self.dic = self.compute()

    def compute_rips_complex_for_subset(self, subset):
        if len(subset) <= 1:
            # Return 0 if the subset contains only one point
            return 0, subset
        else:
            # Compute the maximum distance between any pair of points in the subset
            radius = self.compute_max_distance(subset)
            return radius, subset

    def compute_max_distance(self, subset):
        if not subset:
            # Return 0 if the subset is empty
            return 0

        # Compute the median distance between any pair of points in the subset
        median = sorted([self.distances[i, j] for i in subset for j in subset])[len(subset) // 2]

        # Compute the maximum distance in the subset by comparing the median distance
        # with the maximum distance in the left and right halves of the subset
        left = [self.distances[i, j] for i in subset for j in subset if self.distances[i, j] < median]
        right = [self.distances[i, j] for i in subset for j in subset if self.distances[i, j] > median]
        left_max = max(left) if left else 0
        right_max = max(right) if right else 0

        return max(left_max, right_max)

    def compute(self):
        with Pool() as pool:
            if self.max_size is None:
                # Generate all subsets of the points
                subsets = [subset for k in range(len(self.points)) for subset in
                           combinations(range(len(self.points)), k + 1)]
            else:
                # Generate subsets of the points up to the specified size
                subsets = [subset for k in range(1, self.max_size + 1) for subset in
                           combinations(range(len(self.points)), k + 1)]
            # Compute the maximum distance between any pair of points in each subset
            simplices = pool.map(self.compute_rips_complex_for_subset, subsets)
        # Store the simplices in the faces attribute as tuples instead of arrays
        faces = {tuple(simplex) for radius, simplex in simplices}
        faces.add(tuple())
        # Store the simplices and their corresponding radii in the dic attribute
        dic = {tuple(simplex): radius for radius, simplex in simplices}
        dic[tuple()] = 0

        return faces, dic
