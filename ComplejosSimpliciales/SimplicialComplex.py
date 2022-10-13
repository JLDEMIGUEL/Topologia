from scipy.spatial import Delaunay


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
        return self.order(self.faces)


    def thresholdvalues(self):
            return sorted(list(set(self.dic.values())),key=lambda a: a)

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
        return self.order(set(x for x in self.faces if len(x) - 1 == n))

    def star(self, face):
        """
        star

        Args:
            face: base face of the star

        Returns the star of the given face
        """
        if face not in self.faces:
            return set()
        return self.order(set(x for x in self.faces if set(face).issubset(x)))

    def closedStar(self, face):
        """
        closedStar

        Args:
            face: base face of the closedStar

        Returns the closed star of the given face
        """
        star = self.star(face)
        return self.order(SimplicialComplex(star).faces)

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
        return self.order(lk)

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
        return self.order(skeleton)

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
        vertex = self.skeleton(0)
        vertex.remove(tuple())

        components = set()
        for vert in vertex:
            edges = [x for x in self.face_set() if len(x) == 2]
            components.add(tuple(set(self.reachable(vert[0], edges))))
        return len(components)

    def reachable(self, vert, edges):
        """
        reachable

        Args:
            vert: entry vertex to get the list of reachable vertex
            edges: remaining edges to explore

        Returns list of reachable vertex from the given vertex
        """
        reach = list()
        for edge in edges:
            if vert in edge:
                vert2 = tuple(x for x in edge if x != vert)[0]
                reach.append(vert2)
                edges.remove(edge)
                reach = reach + self.reachable(vert2, edges)
        return reach

    @staticmethod
    def order(faces):
        """
         order

         Args:
             faces: set of faces of a Simplicial Complex

         Returns the ordered list of faces
         """
        # faces.remove(())
        return sorted(faces, key=lambda a: (a, len(a)))
