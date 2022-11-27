import numpy as np

from utils import order, smith_normal_form


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

        orderedFaces = set()
        for x in faces:
            orderedFaces.add(tuple(sorted(list(x), key=lambda a: a)))
        faces = orderedFaces

        self.value = 0
        self.dic = dict()

        self.faces = set()
        for face in faces:
            self.faces.add(face)
            self.subFaces(face)

        self.updateDict(self.faces, 0)

    def add(self, faces, float):
        """
        add

        Args:
            faces: list/set of tuples
        Add the faces to the existing set of faces
        """
        orderedFaces = set()
        for x in faces:
            orderedFaces.add(tuple(sorted(list(x), key=lambda a: a)))
        faces = orderedFaces

        newSC = SimplicialComplex(faces)
        self.faces = self.faces.union(faces)

        self.updateDict(newSC.faces, float)
        return

    def updateDict(self, faces, float):
        """
        updateDict

        Args:
            float:
            faces: list/set of tuples
        Updates de attribute dic with the faces given and the value
        """
        for face in faces:
            if face not in self.dic:
                self.dic[face] = float
            elif self.dic[face] > float:
                self.dic[face] = float
        self.value += 1

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

    def subFaces(self, face):
        """
        subFaces
        Args:
            face: tuple of vertex
        Adds to faces set all the combinations of subFaces
        """
        for vert in face:
            face2 = tuple(x for x in face if x != vert)
            self.faces.add(face2)
            self.subFaces(face2)

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
                reachableList = sorted(self.reachable(vert, visitedVertex), key=lambda a: a)
                components.add(tuple(reachableList))
        return len(components)

    def reachable(self, vert, visitedVertex):
        """
        reachable

        Args:
            visitedVertex: dict with the visited vertex
            vert: entry vertex to get the list of reachable vertex
        Returns list of reachable vertex from the given vertex
        """
        reach = [vert]
        visitedVertex[vert] = True
        for edge in self.n_faces(1):
            if vert in edge:
                tup = tuple(x for x in edge if x != vert)
                endVert = tup[0]
                if not visitedVertex[endVert]:
                    reach = reach + self.reachable(endVert, visitedVertex)
        return reach

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

    def matriz_borde_generalizado(self):
        faces = sorted(self.faces, key=lambda face: (self.dic[face], len(face), face))
        faces.remove(faces[0])
        faces = [(0,), (1,), (2,), (3,), (4,), (5,), (0, 3), (2, 4), (3, 5), (4, 5), (1, 2), (0, 1), (1, 3), (1, 4), (3, 4), (0, 1, 3), (1, 2, 4), (3, 4, 5), (1, 3, 4)]

        M = [[0 for _ in range(len(faces))] for _ in range(len(faces))]
        for i in range(len(faces)):
            for j in range(len(faces)):
                if len(faces[i]) is not len(faces[j]) - 1:
                    continue
                bool = False
                for vert in faces[i]:
                    if vert not in faces[j]:
                        bool = False
                        break
                    bool = True
                if not bool: continue
                M[i][j] = 1
        return M

    def algoritmo_matriz(self, M):
        M = np.matrix(M)
        rows, cols = M.shape
        lows_list = [-1 for _ in range(cols)]
        for j in range(cols):
            columna = M[:, j]
            lista = [x for x in range(rows) if columna[x] == 1]
            if len(lista) == 0: continue
            low = max(lista)
            lows_list[j] = low
            prev_cols = [x for x in range(cols) if lows_list[x] == low and x != j]
            while len(prev_cols) > 0:
                prev_col = prev_cols[0]
                M[:, j] = (M[:, j] + M[:, prev_col]) % 2
                lista = [x for x in range(rows) if M[:, j][x] == 1]
                if len(lista) == 0: break
                low = max(lista)
                lows_list[j] = low
                prev_cols = [x for x in range(cols) if lows_list[x] == low and x != j]
        return M

