def order(faces):
    """
     order

     Args:
         faces: set of faces of a Simplicial Complex

     Returns the ordered list of faces
     """
    # faces.remove(())
    return sorted(faces, key=lambda a: (a, len(a)))


def reachable(edges, vert, visitedVertex):
    """
        reachable

        Args:
            edges: list of edges
            visitedVertex: dict with the visited vertex
            vert: entry vertex to get the list of reachable vertex
        Returns list of reachable vertex from the given vertex
    """
    reach = [vert]
    visitedVertex[vert] = True
    for edge in edges:
        if vert in edge:
            tup = tuple(x for x in edge if x != vert)
            endVert = tup[0]
            if not visitedVertex[endVert]:
                reach = reach + reachable(edges, endVert, visitedVertex)
    return reach


def subFaces(face):
    """
        subFaces
        Args:
            face: tuple of vertex
        Adds to faces set all the combinations of subFaces
        """
    auxSet = set()
    for vert in face:
        face2 = tuple(x for x in face if x != vert)
        auxSet.add(face2)
        auxSet = auxSet.union(subFaces(face2))
    return auxSet


def updateDict(dic, faces, float_value):
    """
    updateDict

    Args:
        dic: dict
        float_value:
        faces: list/set of tuples
    Updates de attribute dic with the faces given and the value
    """
    for face in faces:
        if face not in dic:
            dic[face] = float_value
        elif dic[face] > float_value:
            dic[face] = float_value


def order_faces(faces):
    orderedFaces = set()
    for x in faces:
        orderedFaces.add(tuple(sorted(list(x), key=lambda a: a)))
    faces = orderedFaces
    return faces
