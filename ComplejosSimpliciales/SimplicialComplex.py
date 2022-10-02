class SimplicialComplex:
    """
    Class used to represent a SimplicialComplex

    Attributes:
    
    faces: set
        stores a set of tuples with the vertex of the SimplicialComplex
    dic: dict
        stores a dictionary with faces as keys and float as value

    """

    """
    __init__
    Parameters: 
        faces: list/set of tuples
    """

    def __init__(self, faces):
        self.value = 0
        self.dic = dict()

        self.faces = set()
        for face in faces:
            self.faces.add(face)
            self.subFaces(face)

        self.updateDict(self.faces)

    """
    add

    Parameters: 
        faces: list/set of tuples   
    Add the faces to the existing set of faces
    """

    def add(self, faces):

        newSC = SimplicialComplex(faces)
        self.faces = self.faces.union(newSC.faces)

        self.updateDict(newSC.faces)
        return

    """
    updateDict

    Parameters: 
        faces: list/set of tuples   
    Updates de atribute dic with the faces given and the value
    """

    def updateDict(self, faces):
        for face in faces:
            if face not in self.dic:
                self.dic[face] = self.value
        self.value += 1

    """
    orderByFloat

    Returns a list of faces ordered by their float value and its dimension
    """

    def orderByFloat(self):
        return sorted(self.dic.keys(), key=lambda a: (self.dic[a], len(a)))

    """
    filterByFloat

    Parameters:
        value: Float value 

    Returns a set of faces which float value is less than the given value
    """

    def filterByFloat(self, value):
        keys = self.dic.keys()
        res = {x for x in keys if self.dic[x] <= value}
        return res

    """
    subFaces
    Parameters:
        face: tuple of vertex
    Adds to self.faces set all the combinations of subFaces
    """

    def subFaces(self, face):
        for vert in face:
            face2 = tuple(x for x in face if x != vert)
            self.faces.add(face2)
            self.subFaces(face2)

    """
    face_set

    Returns self.faces
    """

    def face_set(self):
        return self.order(self.faces)

    """
    dimension

    Returns the dimension of the SimplicialComplex
    """

    def dimension(self):
        dim = 1
        for face in self.faces:
            if (dim < len(face)):
                dim = len(face)
        return dim - 1

    """
    n_faces

    Parameters:
        n: dimension

    Returns the faces with dimension n
    """

    def n_faces(self, n):
        return self.order(set(x for x in self.faces if len(x) - 1 == n))

    """
    star

    Parameters:
        face: base face of the star

    Returns the star of the given face
    """

    def star(self, face):
        if face not in self.faces:
            return set()
        return self.order(set(x for x in self.faces if set(face).issubset(x)))

    """
    closedStar

    Parameters:
        face: base face of the closedStar

    Returns the closed star of the given face
    """

    def closedStar(self, face):
        star = self.star(face)
        return self.order(SimplicialComplex(star).faces)

    """
    link

    Parameters:
        face: base face of the link

    Returns the link of the given face
    """

    def link(self, face):
        lk = set()
        for x in self.closedStar(face):
            if len(set(x).intersection(face)) == 0:
                lk.add(x)
        return self.order(lk)

    """
    skeleton

    Parameters:
        dim: dimension of the spected skeleton

    Returns the skeleton with the given dimension
    """

    def skeleton(self, dim):

        skeleton = set()
        for x in self.faces:
            if len(x) <= dim + 1:
                skeleton.add(x)
        return self.order(skeleton)

    """
    euler_characteristic

    Returns the euler characteristic
    """

    def euler_characteristic(self):
        euler = 0
        for i in range(self.dimension() + 1):
            sk = len(set(x for x in self.faces if len(x) == i + 1))
            euler += (-1) ** i * sk
        return euler

    """
    connected_components

    Returns number of connected components of the SimplicialComplex
    """

    def connected_components(self):

        vertex = self.skeleton(0);
        vertex.remove(tuple())

        components = set()
        for vert in vertex:
            edges = [x for x in self.face_set() if len(x) == 2]
            components.add(tuple(set(self.reachable(vert[0], edges))))
        return len(components)

    """
    reachable

    Parameters:
        vert: entry vertex to get the list of reachable vertex
        edges: remaining edges to explore

    Returns list of reachable vertex from the given vertex
    """

    def reachable(self, vert, edges):
        reach = list()
        for edge in edges:
            if (vert in edge):
                vert2 = tuple(x for x in edge if x != vert)[0]
                reach.append(vert2)
                edges.remove(edge)
                reach = reach + self.reachable(vert2, edges)
        return reach

    """
    order

    Parameters:
        faces: set of faces of a Simplicial Complex

    Returns the ordered list of faces
    """

    def order(self, faces):
        # faces.remove(())
        return sorted(faces, key=lambda a: (a, len(a)))
