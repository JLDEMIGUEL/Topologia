import numpy as np

from ComplejosSimpliciales.src.utils.matrices_utils import smith_normal_form
from ComplejosSimpliciales.src.utils.simplicial_complex_utils import order, reachable, subFaces, updateDict, order_faces


class SimplicialComplex:
    """
    Class used to represent a SimplicialComplex

    Attributes:
    
    faces: set
        stores a set of tuples with the vertex of the SimplicialComplex
    dic: dict
        stores a dictionary with faces as keys and float as value

    """

    def __init__(self, faces):
        """
        __init__
        Args:
            faces (list/set): list/set of tuples
        """

        faces = order_faces(faces)

        self.dic = dict()

        self.faces = set()
        for face in faces:
            self.faces.add(face)
            self.faces = self.faces.union(subFaces(face))

        updateDict(self.dic, self.faces, 0)

    def add(self, faces, float_value):
        """
        add

        Args:
            float_value: 
            faces: list/set of tuples
        Add the faces to the existing set of faces
        """
        newSC = SimplicialComplex(faces)
        self.faces = self.faces.union(newSC.face_set())

        updateDict(self.dic, newSC.faces, float_value)

    def orderByFloat(self):
        """
        orderByFloat

        Returns a list of faces ordered by their float value and its dimension
        """
        return sorted(self.dic.keys(), key=lambda a: (self.dic[a], len(a), a))

    def filterByFloat(self, value):
        """
        filterByFloat

        Args:
            value: Float value

        Returns a set of faces which float value is less than the given value
        """
        keys = self.dic.keys()
        res = {x for x in keys if self.dic[x] <= value}
        return res

    def face_set(self):
        """
        face_set

        Returns self.faces
        """
        return order(self.faces)

    def thresholdvalues(self):
        return sorted(list(set(self.dic.values())), key=lambda a: a)

    def dimension(self):
        """
        dimension

        Returns the dimension of the SimplicialComplex
        """
        dim = 1
        for face in self.faces:
            if dim < len(face):
                dim = len(face)
        return dim - 1

    """
    n_faces

    Args:
        n: dimension

    Returns the faces with dimension n
    """

    def n_faces(self, n):
        """
        n_faces

        Args:
            n: dimension

        Returns the faces with dimension n
        """
        return order(set(x for x in self.faces if len(x) - 1 == n))

    def star(self, face):
        """
        star

        Args:
            face: base face of the star

        Returns the star of the given face
        """
        if face not in self.faces:
            return set()
        return order(set(x for x in self.faces if set(face).issubset(x)))

    def closedStar(self, face):
        """
        closedStar

        Args:
            face: base face of the closedStar

        Returns the closed star of the given face
        """
        star = self.star(face)
        return order(SimplicialComplex(star).faces)

    def link(self, face):
        """
        link

        Args:
            face: base face of the link

        Returns the link of the given face
        """
        lk = set()
        for x in self.closedStar(face):
            if len(set(x).intersection(face)) == 0:
                lk.add(x)
        return order(lk)

    def skeleton(self, dim):
        """
        skeleton

        Args:
            dim: dimension of the expected skeleton

        Returns the skeleton with the given dimension
        """
        skeleton = set()
        for x in self.faces:
            if len(x) <= dim + 1:
                skeleton.add(x)
        return order(skeleton)

    def euler_characteristic(self):
        """
        euler_characteristic

        Returns the euler characteristic
        """
        euler = 0
        for i in range(self.dimension() + 1):
            sk = len(set(x for x in self.faces if len(x) == i + 1))
            euler += (-1) ** i * sk
        return euler

    def connected_components(self):
        """
        connected_components

        Returns number of connected components of the SimplicialComplex
        """
        vertex = [x[0] for x in self.n_faces(0)]
        visitedVertex = dict()
        for x in vertex:
            visitedVertex[x] = False
        components = set()
        for vert in vertex:
            if not visitedVertex[vert]:
                reachableList = sorted(reachable(self.n_faces(1), vert, visitedVertex), key=lambda a: a)
                components.add(tuple(reachableList))
        return len(components)

    def boundarymatrix(self, p):
        """
        boundarymatrix

        Args:
            p (int): dimension

        Returns:
            boundary matrix for the given dimension
        """
        Cp = self.n_faces(p)
        Cp_1 = self.n_faces(p - 1)

        Md = [[0 for x in range(len(Cp))] for y in range(len(Cp_1))]

        for i in range(len(Cp_1)):
            for j in range(len(Cp)):
                bool = False
                for vert in Cp_1[i]:
                    if vert not in Cp[j]:
                        bool = False
                        break
                    bool = True
                if not bool: continue
                Md[i][j] = 1
        return Md

    def betti_number(self, p):
        """
        Gets the betti numbers of the simplicial complex for the given dimension p
        Args:
            p: dimension (int)

        Returns the betti_number

        """
        mp = smith_normal_form(np.matrix(self.boundarymatrix(p)))
        mp_1 = smith_normal_form(np.matrix(self.boundarymatrix(p + 1)))
        dim_zp = len([x for x in np.transpose(mp) if 1 not in x])
        dim_bp = len([x for x in mp_1 if 1 in x])
        print('dimzp', dim_zp, 'dimbp', dim_bp, 'bp', dim_zp - dim_bp)
        return dim_zp - dim_bp
