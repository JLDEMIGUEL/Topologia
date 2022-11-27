import numpy as np

from ComplejosSimpliciales.src.SimplicialComplex import SimplicialComplex
from ComplejosSimpliciales.src.utils.vietoris_complex_utils import all_faces, get_all_radios


class Vietoris_RipsComplex(SimplicialComplex):
    """
    Class used to represent a Vietoris_RipsComplex
    Extends from SimplicialComplex

    Attributes:

    points (np.array): stores the complex points
    combinations (set): all faces combinations

    attributes inherited from Vietoris_RipsComplex

    """

    def __init__(self, points: np.array) -> None:
        """
        Args:
            points (np.array): array of points

        Returns:
            None: instantiates new Vietoris_RipsComplex
        """
        self.points = points
        pointPositions = tuple(i for i in range(len(points)))
        self.combinations = all_faces({tuple(x for x in range(len(points)))}, pointPositions)
        self.dic = get_all_radios(self.combinations, self.points)
