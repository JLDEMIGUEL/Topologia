from SimplicialComplex import SimplicialComplex
import math
import numpy as np



class Vietoris_RipsComplex(SimplicialComplex):

    def __init__(self, points):
        self.dic = dict()
        pointsTuple=tuple(x for x in points)
        self.dic[tuple(x for x in range(len(points)))] = self.calcRadio(pointsTuple)
        self.allFaces(pointsTuple)
        print(self.dic)



    def allFaces(self, points):
        for i in range(len(points)):
            face2 = tuple(j for j in range(len(points)) if i != j)
            face3 = tuple(x for x in points if not np.array_equal(x, points[i]))
            self.dic[face2] = self.calcRadio(face3)
            #print(face2)
            self.allFaces(face3)

    def calcRadio(self, points):
        if len(points) <= 1:
            return 0
        maximum = 0
        #print(points)
        for x in points:
            for y in [z for z in points if not np.array_equal(x, z)]:
                #print("hola",x,y)
                dist = math.dist(x, y)
                if maximum < dist:
                    maximum = dist
        return maximum
