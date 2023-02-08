import numpy as np

from SimplicialComplex.utils.matrices_utils import smith_normal_form, generalized_border_matrix_algorithm
from SimplicialComplex.utils.simplicial_complex_utils import order, reachable, sub_faces, updateDict, \
    order_faces, calc_homology, plot_persistence_diagram, plot_barcode_diagram, check_if_sub_face


class SimplicialComplex:
    """
    Class used to represent a SimplicialComplex.

    Attributes:

    faces (dict): stores a dictionary with faces as keys and float as value

    """

    def __init__(self, faces: list | set | tuple) -> None:
        """
        Instantiates new SimplicialComplex.
        Args:
            faces (list | set | tuple): list/set/tuple of tuples with vertex
        Returns:
            None: instantiates new SimplicialComplex
        """
        # Sort the faces vertex lexicographically
        ordered_faces = order_faces(faces)
        # Compute all the faces of the complex
        faces = set()
        for face in ordered_faces:
            faces.add(face)
            faces = faces.union(sub_faces(face))
        # Build the faces dictionary
        self.faces = updateDict({}, faces, 0)

    def add(self, faces: list | set | tuple, float_value: float) -> None:
        """
        Add the faces to the existing set of faces.
        Args:
            float_value (float): value stored in self.faces
            faces (list | set | tuple): list/set of tuples
        Returns:
            None:
        """
        faces = SimplicialComplex(faces).faces_list()
        # Updates the faces dictionary with the new faces
        self.faces = updateDict(self.faces, faces, float_value)

    def filtration_order(self) -> list[tuple]:
        """
        Returns a list of faces ordered by their float value and its dimension.
        Returns:
             list[tuple]: list of faces ordered by their float value
        """
        return sorted(self.faces.keys(), key=lambda a: (self.faces[a], len(a), a))

    def faces_list(self) -> list[tuple]:
        """
        Returns sorted list of faces
        Returns:
             list[tuple]: ordered list of faces
        """
        return order(self.faces.keys())

    def thresholdvalues(self) -> list[int]:
        """
        Returns list of threshold values.
        Returns:
             list[int]: ordered list of threshold values

        """
        return sorted(list(set(self.faces.values())), key=lambda a: a)

    def dimension(self) -> int:
        """
        Returns the dimension of the SimplicialComplex.
        Returns:
            int: dimension
        """
        return len(max(self.faces.keys(), key=len)) - 1

    def n_faces(self, n: int) -> list[tuple]:
        """
        Returns the faces with dimension n.
        Args:
            n (int): dimension
        Returns:
             list[tuple]: faces with dimension n
        """
        return order(set(x for x in self.faces.keys() if len(x) - 1 == n))

    def star(self, face: tuple) -> list[tuple]:
        """
        Computes the star of the given face.
        Args:
            face (tuple): base face of the star
        Returns:
             list[tuple]: star of the given face
        """
        if face not in self.faces.keys():
            return list()
        return order(set(x for x in self.faces.keys() if set(face).issubset(x)))

    def closed_star(self, face: tuple) -> list[tuple]:
        """
        Computes the closed star of the given face.
        Args:
            face (tuple): base face of the closed_star
        Returns:
             list[tuple]: closed star of the given face
        """
        star = self.star(face)
        return order(SimplicialComplex(star).faces.keys())

    def link(self, face: tuple) -> list[tuple]:
        """
        Computes the link of the given face.
        Args:
            face (tuple): base face of the link
        Returns:
            list[tuple]: link of the given face
        """
        link = set()
        for x in self.closed_star(face):
            if len(set(x).intersection(face)) == 0:
                link.add(x)
        return order(link)

    def skeleton(self, dim: int) -> list[tuple]:
        """
        Computes the skeleton of the given dimension.
        Args:
            dim (int): dimension of the expected skeleton
        Returns:
            list[tuple]: skeleton with the given dimension
        """
        skeleton = set()
        for x in self.faces.keys():
            if len(x) <= dim + 1:
                skeleton.add(x)
        return order(skeleton)

    def euler_characteristic(self) -> int:
        """
        Returns the euler characteristic of the complex.
        Returns:
            int: euler characteristic
        """
        euler = 0
        for i in range(self.dimension() + 1):
            sk = len({x for x in self.faces.keys() if len(x) == i + 1})
            euler += (-1) ** i * sk
        return euler

    def connected_components(self) -> int:
        """
        Returns number of connected components of the SimplicialComplex.
        Returns:
            int: number of connected components
        """
        # Build visited vertex dictionary
        vertex, edges = [x[0] for x in self.n_faces(0)], self.n_faces(1)
        visited_vertex = dict()
        for x in vertex:
            visited_vertex[x] = False
        # Compute connected components
        components = set()
        for vert in vertex:
            if not visited_vertex[vert]:
                reachable_list = sorted(reachable(edges, vert, visited_vertex), key=lambda a: a)
                components.add(tuple(reachable_list))
        return len(components)

    def boundary_matrix(self, p: int) -> np.array:
        """
        Returns the boundary matrix of the complex.
        Args:
            p (int): dimension
        Returns:
            np.array: boundary matrix for the given dimension
        """
        Cp = self.n_faces(p)
        Cp_1 = self.n_faces(p - 1)
        Md = [[0 for _ in range(len(Cp))] for _ in range(len(Cp_1))]

        for i in range(len(Cp_1)):
            for j in range(len(Cp)):
                # If is sub-face, add to matrix
                if check_if_sub_face(Cp_1[i], Cp[j]):
                    Md[i][j] = 1
        return np.array(Md)

    def generalized_boundary_matrix(self) -> np.array:
        """
        Computes the generalized border matrix of the complex.
        Returns:
            np.array: the generalized border matrix
        """
        faces = sorted(self.faces.keys(), key=lambda face: (self.faces[face], len(face), face))
        faces.remove(faces[0])
        M = [[0 for _ in range(len(faces))] for _ in range(len(faces))]

        for i in range(len(faces)):
            for j in range(len(faces)):
                # If is sub-face, add to matrix
                if check_if_sub_face(faces[i], faces[j]):
                    M[i][j] = 1
        return np.array(M)

    def betti_number(self, p: int) -> int:
        """
        Gets the betti numbers of the simplicial complex for the given dimension p.
        Args:
            p (int): dimension
        Returns:
            int: betti_number
        """
        mp, _, _ = smith_normal_form(np.array(self.boundary_matrix(p)))
        mp_1, _, _ = smith_normal_form(np.array(self.boundary_matrix(p + 1)))
        # Number of columns of zeros
        dim_zp = len([_ for x in np.transpose(mp) if 1 not in x])
        # Number of rows with ones
        dim_bp = len([_ for x in mp_1 if 1 in x])
        return dim_zp - dim_bp

    def all_betti_numbers(self) -> int:
        """
        Gets the betti numbers of the simplicial complex for the given dimension p.
        Args:
            p (int): dimension
        Returns:
            int: betti_number
        """
        return [self.betti_number(dim) for dim in range(self.dimension())]

    def incremental_algth(self) -> list[int]:
        """
        Gets the betti numbers 0 and 1 of a plain and orientable simplicial complex
        using the incremental algorithm.
        Returns:
            list[int]: betti_number
        """
        faces = self.filtration_order()
        faces.remove(tuple())
        complex_faces = set()
        # Initialize variables to zero
        betti_nums = [0, 0]
        components, loops, triangles = 0, 0, 0

        for face in faces:
            dim = len(face) - 1
            complex_faces.add(face)
            # Compute homology for the current complex and update betti numbers
            c_aux, l_aux, t_aux = calc_homology(complex_faces)
            if c_aux > components or l_aux > loops:
                betti_nums[dim] = betti_nums[dim] + 1
            elif c_aux < components or l_aux < loops:
                betti_nums[dim - 1] = betti_nums[dim - 1] - 1
            if t_aux > triangles:
                betti_nums[dim - 1] = betti_nums[dim - 1] - 1
            components, loops, triangles = c_aux, l_aux, t_aux
        return betti_nums

    def persistence_diagram(self, p: list[int] = None) -> None:
        """
        Computes and plots the persistence diagram of the simplicial complex.
        Args:
             p (list[int]): list of dimensions to plot. If the argument is missing, all the dimensions are used
        Returns:
        """
        infinite, points = self._compute_diagrams_points(p)
        plot_persistence_diagram(points, infinite)

    def barcode_diagram(self,  p: list[int] = None) -> None:
        """
        Computes and plots the barcode diagram of the simplicial complex.
        Args:
             p (list[int]): list of dimensions to plot. If the argument is missing, all the dimensions are used
        Returns:
        """
        infinite, points = self._compute_diagrams_points(p)
        plot_barcode_diagram(points)

    def _compute_diagrams_points(self, p: list[int] = None) -> tuple[int, dict]:
        """
        Computes and plots the barcode diagram of the simplicial complex.
        Args:
            p (list[int]): list of dimensions to plot. If the argument is missing, all the dimensions are used
        Returns:
            tuple[int, dict]: the infinite for the complex and a dictionary mapping dimensions and points to plot
        """
        if p is None:
            p = list(np.array(range(self.dimension())) + 1)
        else:
            p = [p]
        # Compute generalized border matrix and compute the final matrix and lows
        M, lows_list = generalized_border_matrix_algorithm(self.generalized_boundary_matrix())
        faces = sorted(self.faces.keys(), key=lambda _face: (self.faces[_face], len(_face), _face))
        faces.remove(faces[0])
        infinite = 1.5 * max(self.thresholdvalues())
        # Compute points
        points = {}
        for dim in p:
            points_list = set()
            for j, face in enumerate(faces):
                # If face not in lows list, add point j is not a low and dim is correct
                if lows_list[j] == -1:
                    if j not in lows_list and len(face) == dim:
                        points_list.add((self.faces[face], infinite))
                # If face is in lows list, add point if dim is correct
                elif len(face) - 1 == dim:
                    i = lows_list[j]
                    points_list.add((self.faces[faces[i]], self.faces[face]))
            # Add sorted points into points dictionary
            points_list = sorted(points_list, key=lambda point: point[1] - point[0])
            points_list = np.array([np.array(point) for point in points_list])
            points[dim] = points_list
        return infinite, points
