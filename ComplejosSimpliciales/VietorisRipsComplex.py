from SimplicialComplex import SimplicialComplex
import math
import numpy as np



class Vietoris_RipsComplex(SimplicialComplex):

    def __init__(self, points):
        self.dic = dict()
        self.points = points
        self.pointPositions = tuple(i for i in range(len(points)))
        self.combinations = set()
        pointsTuple=tuple(x for x in points)
        self.dic[tuple(x for x in range(len(points)))] = self.calcRadio(pointsTuple)
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
            self.dic[x] = self.calcRadio(tuple(self.points[j] for j in x))

    def calcRadio(self, points):
        if len(points) <= 1:
            return 0
        maximum = 0
        for x in points:
            for y in [z for z in points if not np.array_equal(x, z)]:
                dist = math.dist(x, y)
                if maximum < dist:
                    maximum = dist
        return maximum
