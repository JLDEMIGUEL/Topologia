from itertools import combinations
from multiprocessing import Pool

import numpy as np
from scipy.spatial.distance import pdist, squareform

from SimplicialComplex.SimplicialComplex import SimplicialComplex


class Vietoris_RipsComplex(SimplicialComplex):
    """
    Class used to represent a Vietoris_RipsComplex.

    Attributes:

    points (np.array): stores the complex points
    combinations (set): all faces combinations
    """

    def __init__(self, points: np.array, max_size: int = None) -> None:
        """
        Instantiates a new Vietoris Rips Complex.
        Args:
            points (np.array): array of points
        Returns:
            None: instantiates new Vietoris_RipsComplex
        """
        self.points = points
        self.distances = squareform(pdist(points))
        self.max_size = max_size

        # Compute the Vietoris-Rips complex
        self.faces = self._compute()

    def _compute_rips_complex_for_subset(self, subset: list) -> tuple[int, int]:
        """
        Compute the Vietoris Rips Complex for a given subset of points.
        Args:
            subset (list): A list of points for which to compute the Rips complex
        Returns:
            tuple[int, int]: A tuple containing the radius of the Rips complex, and the subset of points
        """
        if len(subset) <= 1:
            # Return 0 if the subset contains only one point
            return 0, subset
        else:
            # Compute the maximum distance between any pair of points in the subset
            radius = self._compute_max_distance(subset)
            return radius, subset

    def _compute_max_distance(self, subset: list) -> float:
        """
        Compute the maximum distance between any pair of points in a given subset.
        Args:
            subset (list): A list of points for which to compute the maximum distance
        Returns:
            float: The maximum distance between any pair of points in the subset
        """
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

    def _compute(self) -> tuple[set, dict]:
        """
        Compute the Vietoris Rips Complex for the set of points passed to the class.
        Returns:
            tuple[set, dict]: A tuple containing the set of faces of the Rips complex and a dictionary where the keys
            are the faces and the values are the corresponding radii
        """
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
            simplices = pool.map(self._compute_rips_complex_for_subset, subsets)
        # Store the simplices and their corresponding radii in the faces attribute
        dic = {tuple(simplex): radius for radius, simplex in simplices}
        dic[tuple()] = 0

        return dic
