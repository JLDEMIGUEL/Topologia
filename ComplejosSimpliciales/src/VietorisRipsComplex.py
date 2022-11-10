from ComplejosSimpliciales.src.SimplicialComplex import SimplicialComplex
from ComplejosSimpliciales.src.utils.vietoris_complex_utils import calcRadio


class Vietoris_RipsComplex(SimplicialComplex):

    def __init__(self, points):
        self.dic = dict()
        self.points = points
        self.pointPositions = tuple(i for i in range(len(points)))
        self.combinations = set()
        pointsTuple = tuple(x for x in points)
        self.dic[tuple(x for x in range(len(points)))] = calcRadio(pointsTuple)
        self.allFaces(self.pointPositions)
        self.getAllRadios()

    def allFaces(self, points):
        for i in range(len(points)):
            face2 = tuple(j for j in points if i != j)
            sizePrev = self.combinations.__len__()
            self.combinations.add(face2)
            if self.combinations.__len__() == sizePrev:
                return
            self.allFaces(face2)

    def getAllRadios(self):
        for x in list(self.combinations):
            self.dic[x] = calcRadio(tuple(self.points[j] for j in x))
