import numpy as np

from ComplejosSimpliciales.src.SimplicialComplex import SimplicialComplex
from ComplejosSimpliciales.src.utils.vietoris_complex_utils import calcRadio


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
        self.dic = dict()
        self.points = points
        pointPositions = tuple(i for i in range(len(points)))
        self.combinations = set()
        pointsTuple = tuple(x for x in points)
        self.dic[tuple(x for x in range(len(points)))] = calcRadio(pointsTuple)
        self.allFaces(pointPositions)
        self.getAllRadios()

    def allFaces(self, points: tuple) -> None:
        """
        Args:
            points (tuple): tuple of every face

        Returns:
            None: 

        """
        for i in range(len(points)):
            face2 = tuple(j for j in points if i != j)
            sizePrev = self.combinations.__len__()
            self.combinations.add(face2)
            if self.combinations.__len__() == sizePrev:
                return
            self.allFaces(face2)

    def getAllRadios(self) -> None:
        """
        Returns:
            None: 

        """
        for x in list(self.combinations):
            self.dic[x] = calcRadio(tuple(self.points[j] for j in x))
