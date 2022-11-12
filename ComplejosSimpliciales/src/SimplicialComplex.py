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

    def __init__(self, faces: list | set | tuple) -> None:
        """
        __init__
        Args:
            faces (list | set | tuple): list/set of tuples

        Returns:
            None:
        """

        faces = order_faces(faces)

        self.dic = dict()

        self.faces = set()
        for face in faces:
            self.faces.add(face)
            self.faces = self.faces.union(subFaces(face))

        self.dic = updateDict(self.dic, self.faces, 0)

    def add(self, faces: list | set | tuple, float_value: float) -> None:
        """
        add

        Args:
            float_value (float):
            faces (list | set | tuple): list/set of tuples

        Returns:
            None:
        Add the faces to the existing set of faces
        """
        faces = SimplicialComplex(faces).face_set()

        self.faces = self.faces.union(faces)

        self.dic = updateDict(self.dic, faces, float_value)

    def filtration_order(self) -> list[tuple]:
        """
        filtration_order

        Returns a list of faces ordered by their float value and its dimension

        Returns:
             list[tuple]:
        """
        return sorted(self.dic.keys(), key=lambda a: (self.dic[a], len(a), a))

    def face_set(self) -> list[tuple]:
        """
        face_set

        Returns self.faces

        Returns:
             list[tuple]:
        """
        return order(self.faces)

    def thresholdvalues(self) -> list[int]:
        """
        thresholdvalues
        Returns:
             list[int]:

        """
        return sorted(list(set(self.dic.values())), key=lambda a: a)

    def dimension(self) -> int:
        """
        dimension

        Returns the dimension of the SimplicialComplex

        Returns:
            int:
        """
        dim = 1
        for face in self.faces:
            if dim < len(face):
                dim = len(face)
        return dim - 1

    def n_faces(self, n: int) -> list[tuple]:
        """
        n_faces

        Args:
            n (int): dimension

        Returns:
             list[tuple]:

        Returns the faces with dimension n
        """
        return order(set(x for x in self.faces if len(x) - 1 == n))

    def star(self, face: tuple) -> list[tuple]:
        """
        star

        Args:
            face (tuple): base face of the star

        Returns:
             list[tuple]:

        Returns the star of the given face
        """
        if face not in self.faces:
            return list()
        return order(set(x for x in self.faces if set(face).issubset(x)))

    def closedStar(self, face: tuple) -> list[tuple]:
        """
        closedStar

        Args:
            face (tuple): base face of the closedStar

        Returns:
             list[tuple]:

        Returns the closed star of the given face
        """
        star = self.star(face)
        return order(SimplicialComplex(star).faces)

    def link(self, face: tuple) -> list[tuple]:
        """
        link

        Args:
            face (tuple): base face of the link

        Returns:
            list[tuple]:

        Returns the link of the given face
        """
        lk = set()
        for x in self.closedStar(face):
            if len(set(x).intersection(face)) == 0:
                lk.add(x)
        return order(lk)

    def skeleton(self, dim: int) -> list[tuple]:
        """
        skeleton

        Args:
            dim (int): dimension of the expected skeleton

        Returns:
            list[tuple]:

        Returns the skeleton with the given dimension
        """
        skeleton = set()
        for x in self.faces:
            if len(x) <= dim + 1:
                skeleton.add(x)
        return order(skeleton)

    def euler_characteristic(self) -> int:
        """
        euler_characteristic

        Returns the euler characteristic

        Returns:
            int:
        """
        euler = 0
        for i in range(self.dimension() + 1):
            sk = len(set(x for x in self.faces if len(x) == i + 1))
            euler += (-1) ** i * sk
        return euler

    def connected_components(self) -> int:
        """
        connected_components

        Returns number of connected components of the SimplicialComplex

        Returns:
            int:
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

    def boundarymatrix(self, p: int) -> np.matrix:
        """
        boundarymatrix

        Args:
            p (int): dimension

        Returns:
            np.matrix:
            boundary matrix for the given dimension
        """
        Cp = self.n_faces(p)
        Cp_1 = self.n_faces(p - 1)

        Md = [[0 for _ in range(len(Cp))] for _ in range(len(Cp_1))]

        for i in range(len(Cp_1)):
            for j in range(len(Cp)):
                is_in = False
                for vert in Cp_1[i]:
                    if vert not in Cp[j]:
                        is_in = False
                        break
                    is_in = True
                if not is_in:
                    continue
                Md[i][j] = 1
        return np.matrix(Md)

    def betti_number(self, p: int) -> int:
        """
        Gets the betti numbers of the simplicial complex for the given dimension p
        Args:
            p (int): dimension

        Returns:
            int:

        Returns the betti_number

        """
        mp = smith_normal_form(np.matrix(self.boundarymatrix(p)))
        mp_1 = smith_normal_form(np.matrix(self.boundarymatrix(p + 1)))
        dim_zp = len([x for x in np.transpose(mp) if 1 not in x])
        dim_bp = len([x for x in mp_1 if 1 in x])
        return dim_zp - dim_bp
