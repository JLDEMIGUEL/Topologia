from fractions import Fraction
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

from SimplicialComplex.utils.matrices_utils import smith_normal_form, generalized_border_matrix_algorithm, \
    smith_normal_form_z
from SimplicialComplex.utils.simplicial_complex_utils import sort_faces, reachable, sub_faces, updateDict, \
    sort_vertex, calc_homology, plot_persistence_diagram, plot_barcode_diagram, boundary_operator, build_homology_string


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
        faces = set(faces)
        # Compute all the faces of the complex
        for face in faces:
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
        return sort_faces(self.faces.keys())

    def threshold_values(self) -> list[int]:
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
        return sort_faces(filter(lambda x: len(x) - 1 == n, self.faces.keys()))

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
        return sort_faces(set(x for x in self.faces.keys() if set(face).issubset(x)))

    def closed_star(self, face: tuple) -> list[tuple]:
        """
        Computes the closed star of the given face.
        Args:
            face (tuple): base face of the closed_star
        Returns:
             list[tuple]: closed star of the given face
        """
        return sort_faces(SimplicialComplex(self.star(face)).faces.keys())

    def link(self, face: tuple) -> list[tuple]:
        """
        Computes the link of the given face.
        Args:
            face (tuple): base face of the link
        Returns:
            list[tuple]: link of the given face
        """
        return sort_faces(filter(lambda x: len(set(x).intersection(face)) == 0, self.closed_star(face)))

    def skeleton(self, dim: int) -> list[tuple]:
        """
        Computes the skeleton of the given dimension.
        Args:
            dim (int): dimension of the expected skeleton
        Returns:
            list[tuple]: skeleton with the given dimension
        """
        return sort_faces(filter(lambda face: len(face) <= dim + 1, self.faces.keys()))

    def euler_characteristic(self) -> int:
        """
        Returns the euler characteristic of the complex.
        Returns:
            int: euler characteristic
        """
        euler = 0
        for i in range(self.dimension() + 1):
            euler += (-1) ** i * self.betti_number(i, group=2)
        return euler

    def connected_components(self) -> int:
        """
        Returns number of connected components of the SimplicialComplex.
        Returns:
            int: number of connected components
        """
        # Build visited vertex dictionary
        vertex, edges = [x[0] for x in self.n_faces(0)], self.n_faces(1)
        visited_vertex = {x: False for x in vertex}
        # Compute connected components
        components = set()
        for vert in vertex:
            if not visited_vertex[vert]:
                reachable_list = sorted(reachable(edges, vert, visited_vertex), key=lambda a: a)
                components.add(tuple(reachable_list))
        return len(components)

    def boundary_matrix(self, p: int, group: int | str = None) -> np.array:
        """
        Returns the boundary matrix of the complex.
        Args:
            p (int): dimension
            group (int | str): group
        Returns:
            np.array: boundary matrix for the given dimension
        """
        Cp = self.n_faces(p)
        Cp_1 = self.n_faces(p - 1)

        Md = [[boundary_operator(sub_face, super_face) for super_face in Cp] for sub_face in Cp_1]
        if group == 'Q':
            Md = [[Fraction(elem) for elem in row] for row in Md]
        elif group is not None:
            Md = [[boundary_operator(sub_face, super_face) % group for super_face in Cp] for sub_face in Cp_1]
        return np.array(Md)

    def generalized_boundary_matrix(self, group: int | str = None) -> np.array:
        """
        Computes the generalized border matrix of the complex.
        Returns:
            np.array: the generalized border matrix
        """
        faces = sorted(self.faces.keys(), key=lambda face: (self.faces[face], len(face), face))
        faces.remove(faces[0])

        Md = [[boundary_operator(sub_face, super_face) for super_face in faces] for sub_face in faces]
        if group == 'Q':
            Md = [[Fraction(elem) for elem in row] for row in Md]
        elif group is not None:
            Md = [[boundary_operator(sub_face, super_face) % group for super_face in faces] for sub_face in
                  faces]
        return np.array(Md)

    def betti_number(self, p: int, group=None) -> int:
        """
        Gets the betti numbers of the simplicial complex for the given dimension p.
        Args:
            p (int): dimension
            group: group
        Returns:
            int: betti_number
        """
        if group is None:
            mp, _, _, _ = smith_normal_form_z(self.boundary_matrix(p, group))
            mp_1, _, _, _ = smith_normal_form_z(self.boundary_matrix(p + 1, group))
        else:
            mp, _, _, _ = smith_normal_form(self.boundary_matrix(p, group), group=group)
            mp_1, _, _, _ = smith_normal_form(self.boundary_matrix(p + 1, group), group=group)
        # Number of columns of zeros
        dim_zp = len([_ for x in np.transpose(mp) if sum(x) == 0])
        # Number of rows with ones
        dim_bp = len([_ for x in mp_1 if sum(x) != 0])
        return dim_zp - dim_bp

    def all_betti_numbers(self, group=None) -> int:
        """
        Gets the betti numbers of the simplicial complex for the given dimension p.
        Args:
            group: group
        Returns:
            int: betti_number
        """
        return [self.betti_number(dim, group) for dim in range(self.dimension())]

    def homology(self, p: int, group: int | str = None) -> str:
        """
        Computes the homology of the simplicial complex up to degree p, with coefficients in the given group.

        Args:
            p (int): The degree up to which to compute the homology. The homology groups computed will have degree up to p.
            group (optional): The coefficients to use in homology computations. If None, uses the integers (Z) as coefficients.

        Returns:
            str: A string describing the homology groups of the simplicial complex up to degree p, with coefficients in the given group.
        """
        if group is None:
            mp_1, _, _, _ = smith_normal_form_z(self.boundary_matrix(p + 1, group))
        else:
            mp_1, _, _, _ = smith_normal_form(self.boundary_matrix(p + 1, group), group=group)

        betti = self.betti_number(p, group)

        return build_homology_string(betti, group, mp_1)

    def cohomology(self, p: int, group: int | str = None) -> str:
        """
        Computes the cohomology of the simplicial complex up to degree p, with coefficients in the given group.

        Args:
            p (int): The degree up to which to compute the cohomology. The homology groups computed will have degree up to p.
            group (optional): The coefficients to use in cohomology computations. If None, uses the integers (Z) as coefficients.

        Returns:
            str: A string describing the cohomology groups of the simplicial complex up to degree p, with coefficients in the given group.
        """
        if group is None:
            mp_1, _, _, _ = smith_normal_form_z(self.boundary_matrix(p, group))
        else:
            mp_1, _, _, _ = smith_normal_form(self.boundary_matrix(p, group), group=group)

        betti = self.betti_number(p, group)

        return build_homology_string(betti, group, mp_1)

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
        plt.show()

    def barcode_diagram(self, p: list[int] = None) -> None:
        """
        Computes and plots the barcode diagram of the simplicial complex.
        Args:
             p (list[int]): list of dimensions to plot. If the argument is missing, all the dimensions are used
        Returns:
        """
        infinite, points = self._compute_diagrams_points(p)
        plot_barcode_diagram(points)
        plt.show()

    def png_persistence_diagram(self, p: list[int] = None) -> None:
        """
        Computes and plots the persistence diagram of the simplicial complex.
        Args:
             p (list[int]): list of dimensions to plot. If the argument is missing, all the dimensions are used
        Returns:
        """
        infinite, points = self._compute_diagrams_points(p)
        fig = plt.figure()
        plot_persistence_diagram(points, infinite)
        plt.close()
        png_output = BytesIO()
        fig.savefig(png_output, format='png')
        png_output.seek(0)
        return png_output.getvalue()

    def png_barcode_diagram(self, p: list[int] = None) -> None:
        """
        Computes and plots the barcode diagram of the simplicial complex.
        Args:
             p (list[int]): list of dimensions to plot. If the argument is missing, all the dimensions are used
        Returns:
        """
        infinite, points = self._compute_diagrams_points(p)
        fig = plt.figure()
        plot_barcode_diagram(points)
        plt.close()
        png_output = BytesIO()
        fig.savefig(png_output, format='png')
        png_output.seek(0)
        return png_output.getvalue()

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
        M, lows_list = generalized_border_matrix_algorithm(self.generalized_boundary_matrix(group=2))
        faces = sorted(self.faces.keys(), key=lambda _face: (self.faces[_face], len(_face), _face))
        faces.remove(faces[0])
        infinite = 1.5 * max(self.threshold_values())
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
