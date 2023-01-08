import numpy as np
from matplotlib import pyplot as plt

from ComplejosSimpliciales.src.utils.matrices_utils import smith_normal_form, algoritmo_matriz, \
    matriz_borde_generalizado
from ComplejosSimpliciales.src.utils.simplicial_complex_utils import order, reachable, subFaces, updateDict, \
    order_faces, connected_components, num_loops, num_triang


class SimplicialComplex:
    """
    Class used to represent a SimplicialComplex

    Attributes:
    
    faces (set): stores a set of tuples with the vertex of the SimplicialComplex
    dic (dict): stores a dictionary with faces as keys and float as value

    """

    def __init__(self, faces: list | set | tuple) -> None:
        """
        Instantiates new SimplicialComplex

        Args:
            faces (list | set | tuple): list/set/tuple of tuples with vertex

        Returns:
            None: instantiates new SimplicialComplex
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
        Add the faces to the existing set of faces

        Args:
            float_value (float): value stored in self.dic
            faces (list | set | tuple): list/set of tuples
        Returns:
            None:
        """
        faces = SimplicialComplex(faces).face_set()

        self.faces = self.faces.union(faces)

        self.dic = updateDict(self.dic, faces, float_value)

    def filtration_order(self) -> list[tuple]:
        """
        Returns a list of faces ordered by their float value and its dimension

        Returns:
             list[tuple]: list of faces ordered by their float value
        """
        return sorted(self.dic.keys(), key=lambda a: (self.dic[a], len(a), a))

    def face_set(self) -> list[tuple]:
        """
        Returns self.faces

        Returns:
             list[tuple]: ordered list of self.faces
        """
        return order(self.faces)

    def thresholdvalues(self) -> list[int]:
        """
        Returns list of threshold values

        Returns:
             list[int]: ordered list of threshold values

        """
        return sorted(list(set(self.dic.values())), key=lambda a: a)

    def dimension(self) -> int:
        """
        Returns the dimension of the SimplicialComplex

        Returns:
            int: dimension
        """
        dim = 1
        for face in self.faces:
            if dim < len(face):
                dim = len(face)
        return dim - 1

    def n_faces(self, n: int) -> list[tuple]:
        """
        Args:
            n (int): dimension

        Returns:
             list[tuple]: faces with dimension n
        """
        return order(set(x for x in self.faces if len(x) - 1 == n))

    def star(self, face: tuple) -> list[tuple]:
        """
        Args:
            face (tuple): base face of the star

        Returns:
             list[tuple]: star of the given face
        """
        if face not in self.faces:
            return list()
        return order(set(x for x in self.faces if set(face).issubset(x)))

    def closedStar(self, face: tuple) -> list[tuple]:
        """
        Args:
            face (tuple): base face of the closedStar

        Returns:
             list[tuple]: closed star of the given face
        """
        star = self.star(face)
        return order(SimplicialComplex(star).faces)

    def link(self, face: tuple) -> list[tuple]:
        """
        Args:
            face (tuple): base face of the link

        Returns:
            list[tuple]: link of the given face
        """
        lk = set()
        for x in self.closedStar(face):
            if len(set(x).intersection(face)) == 0:
                lk.add(x)
        return order(lk)

    def skeleton(self, dim: int) -> list[tuple]:
        """
        Args:
            dim (int): dimension of the expected skeleton

        Returns:
            list[tuple]: skeleton with the given dimension
        """
        skeleton = set()
        for x in self.faces:
            if len(x) <= dim + 1:
                skeleton.add(x)
        return order(skeleton)

    def euler_characteristic(self) -> int:
        """
        Returns the euler characteristic

        Returns:
            int: euler characteristic
        """
        euler = 0
        for i in range(self.dimension() + 1):
            sk = len(set(x for x in self.faces if len(x) == i + 1))
            euler += (-1) ** i * sk
        return euler

    def connected_components(self) -> int:
        """
        Returns number of connected components of the SimplicialComplex

        Returns:
            int: number of connected components
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
        Args:
            p (int): dimension

        Returns:
            np.matrix: boundary matrix for the given dimension
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
            int: betti_number

        """
        mp = smith_normal_form(np.matrix(self.boundarymatrix(p)))
        mp_1 = smith_normal_form(np.matrix(self.boundarymatrix(p + 1)))
        dim_zp = len([x for x in np.transpose(mp) if 1 not in x])
        dim_bp = len([x for x in mp_1 if 1 in x])
        return dim_zp - dim_bp

    def algoritmo_incremental(self):
        faces = self.filtration_order()
        faces.remove(tuple())
        complex = set()
        betti_nums = [0, 0]
        components, loops, triangles = 0, 0, 0
        for face in faces:
            dim = len(face) - 1
            complex.add(face)
            c_aux, l_aux, t_aux = self.calc_homology(complex)
            if c_aux > components or l_aux > loops:
                betti_nums[dim] = betti_nums[dim] + 1
            elif c_aux < components or l_aux < loops:
                betti_nums[dim - 1] = betti_nums[dim - 1] - 1
            if t_aux > triangles:
                betti_nums[dim - 1] = betti_nums[dim - 1] - 1
            components, loops, triangles = c_aux, l_aux, t_aux
        return betti_nums
    def calc_homology(self, complex):
        return self.calc_comps(complex), self.calc_loops(complex), self.calc_triang(complex)

    def calc_comps(self, complex):
        return connected_components(complex)

    def calc_loops(self, complex):
        return num_loops(complex)

    def calc_triang(self, complex):
        return num_triang(complex)

    def persistence_diagram(self, p=None):
        if p is None:
            p = list(np.array(range(self.dimension())) + 1)
        else:
            p = [p]
        M, lows_list = algoritmo_matriz(matriz_borde_generalizado(self.dic))
        faces = sorted(self.faces, key=lambda face: (self.dic[face], len(face), face))
        faces.remove(faces[0])
        infinite = 1.5 * max(self.thresholdvalues())

        for dim in p:
            points = []
            for j in range(len(faces)):
                if lows_list[j] == -1:
                    if j not in lows_list and len(faces[j]) == dim:
                        points.append((self.dic[faces[j]], infinite))
                elif len(faces[j]) - 1 == dim:
                    i = lows_list[j]
                    points.append((self.dic[faces[i]], self.dic[faces[j]]))
            self.plot_points(points, dim)
        plt.axis([-0.1 * infinite, infinite * 1.1, -0.1 * infinite, infinite * 1.1])
        plt.plot([-0.1 * infinite, infinite * 1.1], [-0.1 * infinite, infinite * 1.1], "b--")
        plt.plot([-0.1 * infinite, infinite * 1.1], [infinite, infinite], "b--")
        plt.show()

    def plot_points(self, points, dim):
        colors = ["b", "g", "r", "m", "y"]
        points = np.array([np.array(point) for point in points])
        plt.plot(points[:, 0], points[:, 1], colors[dim % len(colors)] + "o")

    def barcode_diagram(self):
        M, lows_list = algoritmo_matriz(matriz_borde_generalizado(self.dic))
        faces = sorted(self.faces, key=lambda face: (self.dic[face], len(face), face))
        faces.remove(faces[0])
        infinite = 1.5 * max(self.thresholdvalues())
        p = list(np.array(range(self.dimension())) + 1)
        h = 0
        colors = ["b", "g", "r", "m", "y"]
        for dim in p:
            points = set()
            for j in range(len(faces)):
                if lows_list[j] == -1:
                    if j not in lows_list and len(faces[j]) == dim:
                        points.add((self.dic[faces[j]], infinite))
                elif len(faces[j]) - 1 == dim:
                    i = lows_list[j]
                    points.add((self.dic[faces[i]], self.dic[faces[j]]))
            points = sorted(points, key=lambda point: point[1] - point[0])
            points = np.array([np.array(point) for point in points])
            for i in range(len(points)):
                plt.plot([points[i][0], points[i][1]], [h, h], colors[dim])
                h += 1
        plt.show()
